import gymnasium as gym
import os
import numpy as np
import torch
from torch import nn
import imageio

from tqdm import tqdm
import pickle

from diffusion_policy_transformer import PolicyDiffusionTransformer
from PIL import Image
from diffusers import DDPMScheduler, DDIMScheduler
import time
import json
import argparse
import matplotlib.pyplot as plt
from train_dagger_BC import make_trial_dir # since I don't want to implement it again

try:
    import wandb
except ImportError:
    wandb = None

class TrainDiffusionPolicy:

    def __init__(
        self,
        env,
        model, 
        optimizer, 
        states_array, 
        actions_array, 
        device="cpu", 
        num_train_diffusion_timesteps=30,
        max_trajectory_length=1600,
    ):
        """
        Initializes the TrainDiffusionPolicy class. Creates necessary data structures and normalizes states AND actions.

        Args:
            env (gym.Env): The environment that the model is trained on.
            model (PolicyDiffusionTransformer): the model to train
            optimizer (torch.optim.Optimizer): the optimizer to use for training the model
            states_array (np.ndarray): the states to train on
            actions_array (np.ndarray): the actions to train on
            device (str): the device to use for training
            num_train_diffusion_timesteps (int): the number of diffusion timesteps to use for training
        """
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.states = states_array
        self.actions = actions_array

        self.action_dimension = self.actions.shape[-1]
        self.state_dimension = self.states.shape[-1]

        # clip all actions to be between -1 and 1, as this is the range that the environment expects
        self.clip_sample_range = 1
        self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        self.trajectory_lengths = np.any(self.states != 0, axis=-1).sum(axis=1) # HW ERRATA 5
        self.max_trajectory_length = max_trajectory_length

        model.set_device(self.device)

        # normalize states and actions
        all_states = np.concatenate([self.states[i, 0:self.trajectory_lengths[i]] for i in range(len(self.states))], axis=0)
        all_actions = np.concatenate([self.actions[i, 0:self.trajectory_lengths[i]] for i in range(len(self.actions))], axis=0)

        self.states_mean = np.mean(all_states, axis=(0))
        self.states_std = np.std(all_states, axis=(0))
        self.states = (self.states - self.states_mean) / self.states_std

        self.actions_mean = np.mean(all_actions, axis=(0))
        self.actions_std = np.std(all_actions, axis=(0))
        self.actions = (self.actions - self.actions_mean) / self.actions_std

        self.num_train_diffusion_timesteps = num_train_diffusion_timesteps

        # training and inference schedulers for diffusion
        self.training_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=False, # HW ERRATA 4
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log", # variance is different for inference, see paper https://arxiv.org/pdf/2301.10677
            clip_sample=False, # HW ERRATA 4
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler.alphas_cumprod = self.inference_scheduler.alphas_cumprod.to(self.device)


    def get_inference_timesteps(self):
        """
        gets the timesteps to use for inference
        """
        self.inference_scheduler.set_timesteps(self.num_train_diffusion_timesteps, device=self.device)
        return self.inference_scheduler.timesteps

    def diffusion_sample(
        self,
        previous_states, 
        previous_actions,
        episode_timesteps,
        previous_states_padding_mask=None,
        previous_actions_padding_mask=None,
        actions_padding_mask=None,
        max_action_len=3,
    ):
        """
        perform a single diffusion sample from noise to actions

        Args:
            previous_states (torch.Tensor): the previous states to condition on
            previous_actions (torch.Tensor): the previous actions to condition on
            episode_timesteps (torch.Tensor): the episode timesteps to condition on
            previous_states_padding_mask (torch.Tensor): the padding mask for the previous states
            previous_actions_padding_mask (torch.Tensor): the padding mask for the previous actions
            actions_padding_mask (torch.Tensor): the padding mask for the actions being predicted
            max_action_len (int): the maximum number of actions to predict

        NOTE: remember that you are predicting max_action_len actions, not just one
        """
        # BEGIN STUDENT SOLUTION

        batch_size = previous_states.shape[0]

        with torch.no_grad():
            predicted_actions = torch.randn((batch_size, max_action_len, self.action_dimension), device=self.device)

            for t in self.get_inference_timesteps():
                noise_levels = torch.full((batch_size, 1), int(t), dtype=torch.long, device=self.device)
                predicted_noise = self.model(
                    previous_states,
                    previous_actions,
                    predicted_actions,
                    episode_timesteps,
                    noise_levels,
                    previous_states_padding_mask,
                    previous_actions_padding_mask,
                    actions_padding_mask,
                )

                predicted_actions = self.inference_scheduler.step(
                    predicted_noise, 
                    t, 
                    predicted_actions
                ).prev_sample

        # END STUDENT SOLUTION
        return predicted_actions

    def sample_trajectory(
        self, 
        env, 
        num_actions_to_eval_in_a_row=3, 
        num_previous_states=5,
        num_previous_actions=4, 
        render=False,
    ):
        """
        run a trajectory using the trained model

        Args:
            env (gym.Env): the environment to run the trajectory in
            num_actions_to_eval_in_a_row (int): the number of actions to evaluate in a row
            num_previous_states (int): the number of previous states to condition on
            num_previous_actions (int): the number of previous actions to condition on
            render (bool): whether to save images from environment

        NOTE: use with torch.no_grad(): to speed up inference by not storing gradients
        NOTE: for the first few steps, make sure to add padding to previous states/actions - use False if a state/action should be included, and True if it should be padded
        NOTE: both states and actions should be normalized before being passed to the model, and the model outputs normalized actions that need to be denormalized
        NOTE: refer to the forward function of diffusion_policy_transformer to see how to pass in the inputs (tensor shapes, etc.)
        """
        rewards, rgbs = np.zeros((self.max_trajectory_length,)), []
        # BEGIN STUDENT SOLUTION

        def normalize(data, mean, std):
            return (np.array(data) - mean) / std

        def unnormalize(data, mean, std):
            return data * std + mean

        def pad_and_mask(data_list, max_len, dim, mean, std):
            num_items = len(data_list)
            padded_array = np.zeros((max_len, dim))
            
            if num_items > 0:
                padded_array[:num_items] = normalize(data_list, mean, std)
                
            mask = np.zeros(max_len, dtype=bool)
            mask[num_items:] = True
            
            return padded_array, mask

        state, _ = env.reset()
        states_list, actions_list, timesteps_list = [state], [], [0]

        if render:
            rgbs.append(env.render())

        done, truncated = False, False

        while not done and not truncated:
            states_list = states_list[-num_previous_states:]
            actions_list = actions_list[-num_previous_actions:]
            timesteps_list = timesteps_list[-num_previous_states:]

            model_input_previous_states, previous_states_padding_mask = pad_and_mask(
                states_list, num_previous_states, self.state_dimension, self.states_mean, self.states_std
            )
            model_input_previous_actions, previous_actions_padding_mask = pad_and_mask(
                actions_list, num_previous_actions, self.action_dimension, self.actions_mean, self.actions_std
            )

            num_states = len(states_list)
            episode_timesteps = np.arange(timesteps_list[0], timesteps_list[0] + num_previous_states)

            predicted_actions_raw = self.diffusion_sample(
                torch.from_numpy(model_input_previous_states).float().unsqueeze(0).to(self.device),
                torch.from_numpy(model_input_previous_actions).float().unsqueeze(0).to(self.device),
                torch.from_numpy(episode_timesteps).long().unsqueeze(0).to(self.device),
                torch.from_numpy(previous_states_padding_mask).unsqueeze(0).to(self.device),
                torch.from_numpy(previous_actions_padding_mask).unsqueeze(0).to(self.device),
            )

            predicted_actions_unclipped = unnormalize(
                predicted_actions_raw[0].cpu().numpy(), 
                self.actions_mean, 
                self.actions_std
            )
            predicted_actions = np.clip(
                predicted_actions_unclipped, 
                -self.clip_sample_range, 
                self.clip_sample_range
            )

            for action in predicted_actions[:num_actions_to_eval_in_a_row]:
                state, reward, done, truncated, _ = env.step(action)
                rewards[timesteps_list[-1]] = reward

                if render:
                    rgbs.append(env.render())

                if done or truncated:
                    break

                states_list.append(state)
                actions_list.append(action)
                timesteps_list.append(timesteps_list[-1] + 1)


        # END STUDENT SOLUTION
        return rewards, rgbs

    def evaluation(
        self,
        diffusion_policy_iter=None, 
        num_samples=20,
        num_actions_to_eval_in_a_row=3,
        model_dir="data/diffusion_policy_transformer_models",
        output_dir=None,
    ):
        """
        evaluate the model on the environment

        Args:
            diffusion_policy_iter (Optional[int]): the iteration to load the diffusion policy from
            num_samples (int): the number of samples to evaluate
            num_actions_to_eval_in_a_row (int): the number of predicted actions to execute before resampling
            model_dir (str): the directory to load the model weights from
            output_dir (Optional[str]): the directory to save per-trajectory rewards and times to

        NOTE: feel free to change this function when making graphs
        """
        # load model weights:
        if diffusion_policy_iter is None:
            self.model.load_state_dict(torch.load(os.path.join(model_dir, "diffusion_policy.pt"), map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(os.path.join(model_dir, f"diffusion_policy_iter_{diffusion_policy_iter}.pt"), map_location=self.device))

 
        self.model.eval() # turn on eval mode (this turns off dropout, running_mean, etc. that are used in training)

        rewards = np.zeros((num_samples, self.max_trajectory_length))
        times = np.zeros(num_samples)
        for sample_trajectory in tqdm(range(num_samples)):
            time1 = time.time()
            reward, _ = self.sample_trajectory(self.env, num_actions_to_eval_in_a_row=num_actions_to_eval_in_a_row)
            time2 = time.time()
            times[sample_trajectory] = time2 - time1
            print(f"trajectory {sample_trajectory} took {time2 - time1} seconds")
            rewards[sample_trajectory] = reward
            print(f"rewards from trajectory {sample_trajectory}={reward.sum()}")
        print(f"average reward per trajectory={rewards.sum() / (rewards.shape[0])}")
        print(f"median reward per trajectory={np.median(rewards.sum(axis=1))}")
        print(f"max reward per trajectory={np.max(rewards.sum(axis=1))}")
        print(f"average trajectory length={np.mean(np.array([sum(1 for r in rewards[i] if r != 0) for i in range(len(rewards))]))}")
        print(f"average time per trajectory={times.mean()}")

        trajectory_rewards = rewards.sum(axis=1)
        stats = {
            "num_actions_to_eval_in_a_row": num_actions_to_eval_in_a_row,
            "num_samples": num_samples,
            "avg_trajectory_time": float(times.mean()),
            "mean_reward": float(trajectory_rewards.mean()),
            "median_reward": float(np.median(trajectory_rewards)),
            "max_reward": float(trajectory_rewards.max()),
            "avg_trajectory_length": float(np.mean(np.count_nonzero(rewards, axis=1))),
        }
        if output_dir is not None:
            np.savez(
                os.path.join(output_dir, f"eval_{num_actions_to_eval_in_a_row}_actions.npz"),
                rewards=rewards,
                times=times,
            )
        return stats

    def train(
        self, 
        num_training_steps, 
        batch_size=64, 
        print_every=5000, 
        save_every=10000, 
        wandb_logging=False,
        output_dir="data/diffusion_policy_transformer_models",
    ):
        """
        training loop that calls training_step

        Args:
            num_training_steps (int): the number of training steps to run
            batch_size (int): the batch size to use
            print_every (int): how often to print the loss
            save_every (int): how often to save the model
            wandb_logging (bool): whether to log to wandb
            output_dir (str): the directory to save checkpoints, the loss plot, and wandb runs to
        """
        model = self.model
        os.makedirs(output_dir, exist_ok=True)
        if wandb_logging:
            wandb.init(
                name="diffusion transfomer training",
                group="diffuson transformer",
                project='DeepRL HW4 Q2',
                dir=output_dir,
            )

        losses = np.zeros(num_training_steps)
        model.train()
        for training_iter in tqdm(range(num_training_steps)):
            loss = self.training_step(batch_size)
            losses[training_iter] = loss
            if wandb_logging:
                wandb.log({"loss": loss}, step=training_iter)
            if training_iter % print_every == 0:
                print(f"Training Iteration {training_iter}: loss = {loss}")
            if (training_iter + 1) % save_every == 0:
                torch.save(model.state_dict(), os.path.join(output_dir, f"diffusion_policy_iter_{training_iter + 1}.pt"))

        torch.save(model.state_dict(), os.path.join(output_dir, "diffusion_policy.pt"))
        if wandb_logging:
            wandb.finish()

        x_axis = np.arange(num_training_steps)
        plt.figure()
        plt.plot(x_axis, losses)
        plt.xlabel("Training Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Diffusion Policy")
        plt.savefig(os.path.join(output_dir, "diffusion_policy_loss.png"))
        np.savez(os.path.join(output_dir, "diffusion_policy_losses.npz"), losses=losses)
        print(f"final loss={losses[-1]}")

        return losses

    def training_step(self, batch_size):
        """
        Runs a single training step on the model.

        Args:
            batch_size (int): The batch size to use.

        NOTE: actions_padding is a mask that is False for actions to be predicted and True otherwise 
                (for instance, the model predicts 3 actions, but our batch element may contain the 2 final actions in a sequence)
                when calculating the loss, we should only consider the loss for the actions that are not padded
        NOTE: return a loss value that is a plain float (not a tensor), and is on cpu
        """
        # BEGIN STUDENT SOLUTION


        (
            previous_states,
            previous_actions,
            actions,
            episode_timesteps,
            previous_states_padding_mask,
            previous_actions_padding_mask,
            actions_padding_mask,
        ) = self.get_training_batch(batch_size)

        noise = torch.randn_like(actions)
        noise_timesteps = torch.randint(0, self.num_train_diffusion_timesteps, (actions.shape[0],), device=self.device)
        noisy_actions = self.training_scheduler.add_noise(actions, noise, noise_timesteps)

        predicted_noise = self.model(
            previous_states,
            previous_actions,
            noisy_actions,
            episode_timesteps,
            noise_timesteps.unsqueeze(1),
            previous_states_padding_mask,
            previous_actions_padding_mask,
            actions_padding_mask,
        )

        loss = nn.functional.mse_loss(predicted_noise[~actions_padding_mask], noise[~actions_padding_mask])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = loss.detach().cpu().item()

        # END STUDENT SOLUTION

        return loss

    def get_training_batch(self, batch_size, max_action_len=3, num_previous_states=5, num_previous_actions=4):
        """
        get a training batch for the model
        Args:
            batch_size (int): the batch size to use
            max_action_len (int): the maximum number of actions to predict
            num_previous_states (int): the number of previous states to condition on
            num_previous_actions (int): the number of previous actions to condition on
        """
        assert num_previous_states == num_previous_actions + 1, f"num_previous_states={num_previous_states} must be equal to num_previous_actions + 1={num_previous_actions + 1}"

        # get trajectory lengths, so we can sample each trajectory with probability proportional to its length
        # this is equivalent to sampling uniformly from the set of all environment steps
        batch_indices = np.random.choice(
            np.arange(len(self.trajectory_lengths)),
            size=batch_size,
            replace=True,
            p=np.array(self.trajectory_lengths) / sum(self.trajectory_lengths)
        )

        previous_states_batch, previous_actions_batch, actions_batch, episode_timesteps_batch, previous_states_padding_batch, previous_actions_padding_batch, actions_padding_batch = [], [], [], [], [], [], []
        for i in range(len(batch_indices)):
            # get the start and end index for states to condition on
            end_index_state = np.random.randint(1, self.trajectory_lengths[batch_indices[i]] + 1) # HW ERRATA 3
            start_index_state = max(0, end_index_state - num_previous_states)

            # get the start and end index for actions to condition on (we predict the action for the final state)
            start_index_previous_actions = start_index_state
            end_index_previous_actions = end_index_state - 1

            # get the start and end index for actions to predict
            start_index_action = end_index_previous_actions # HW ERRATA 2
            end_index_action = min(self.trajectory_lengths[batch_indices[i]], start_index_action + max_action_len)

            previous_states = self.states[batch_indices[i], start_index_state:end_index_state]
            previous_actions = self.actions[batch_indices[i], start_index_previous_actions:end_index_previous_actions]
            actions = self.actions[batch_indices[i], start_index_action:end_index_action]

            state_dim = previous_states.shape[1]
            action_dim = actions.shape[1]

            state_seq_length = previous_states.shape[0]
            previous_action_seq_length = previous_actions.shape[0]

            # if we have less than the max number of previous states, add some padding (i.e. we're predicting a very early state)
            if state_seq_length < num_previous_states:
                previous_states = np.concatenate([previous_states, np.zeros((num_previous_states - state_seq_length, state_dim))], axis=0)
                previous_actions = np.concatenate([previous_actions, np.zeros((num_previous_actions - previous_action_seq_length, action_dim))], axis=0)
                previous_states_padding_mask = np.concatenate([np.zeros(state_seq_length), np.ones(num_previous_states - state_seq_length)], axis=0)
                previous_actions_padding_mask = np.concatenate([np.zeros(previous_action_seq_length), np.ones(num_previous_actions - previous_action_seq_length)], axis=0)
            else:
                previous_states_padding_mask = np.zeros(num_previous_states)
                previous_actions_padding_mask = np.zeros(num_previous_actions)

            # if we have less than the max number of actions, add some padding (i.e. we're predicting a very early action)
            action_seq_length = actions.shape[0]
            if action_seq_length < max_action_len:
                action_dim = actions.shape[1]
                actions = np.concatenate([actions, np.zeros((max_action_len - action_seq_length, action_dim))], axis=0)
                action_padding_mask = np.concatenate([np.zeros(action_seq_length), np.ones(max_action_len - action_seq_length)], axis=0)

            else:
                action_padding_mask = np.zeros(max_action_len)

            previous_states_batch.append(previous_states)
            previous_actions_batch.append(previous_actions)
            actions_batch.append(actions)
            episode_timesteps_batch.append(np.arange(start_index_state, start_index_state + num_previous_states)) # add extra dummy timesteps in some cases
            previous_states_padding_batch.append(previous_states_padding_mask)
            previous_actions_padding_batch.append(previous_actions_padding_mask)
            actions_padding_batch.append(action_padding_mask)

        previous_states_batch = np.stack(previous_states_batch)
        previous_actions_batch = np.stack(previous_actions_batch)
        actions_batch = np.stack(actions_batch)
        episode_timesteps_batch = np.stack(episode_timesteps_batch)
        previous_states_padding_batch = np.stack(previous_states_padding_batch)
        previous_actions_padding_batch = np.stack(previous_actions_padding_batch)
        actions_padding_batch = np.stack(actions_padding_batch)

        previous_states_batch = torch.from_numpy(previous_states_batch).float().to(self.device)
        previous_actions_batch = torch.from_numpy(previous_actions_batch).float().to(self.device)
        actions_batch = torch.from_numpy(actions_batch).float().to(self.device)
        previous_states_padding_batch = torch.from_numpy(previous_states_padding_batch).bool().to(self.device)
        previous_actions_padding_batch = torch.from_numpy(previous_actions_padding_batch).bool().to(self.device)
        actions_padding_batch = torch.from_numpy(actions_padding_batch).bool().to(self.device)
        episode_timesteps_batch = torch.from_numpy(episode_timesteps_batch).long().to(self.device)

        return previous_states_batch, previous_actions_batch, actions_batch, episode_timesteps_batch, previous_states_padding_batch, previous_actions_padding_batch, actions_padding_batch

def run_training(
    output_dir=None,
    num_training_steps=50000,
    skip_training=False,
    eval_samples=20,
    eval_actions_in_a_row=(3, 2, 1),
    max_gif_attempts=None,
):
    """
    Creates the environment, model, and optimizer, loads the data, and trains/evaluates the model using the TrainDiffusionPolicy class.

    Args:
        output_dir (Optional[str]): where to save results, defaults to the next unused results_diffusion/trial_{number}
        num_training_steps (int): the number of training steps to run
        skip_training (bool): evaluate the model already saved in output_dir instead of training a new one
        eval_samples (int): the number of trajectories to evaluate for each num_actions_to_eval_in_a_row
        eval_actions_in_a_row (tuple): the values of num_actions_to_eval_in_a_row to evaluate
        max_gif_attempts (Optional[int]): stop looking for a successful gif after this many trajectories
    """

    if output_dir is None:
        output_dir = make_trial_dir("diffusion")
    else:
        os.makedirs(output_dir, exist_ok=True)
    print(f"saving results to {output_dir}")

    env = gym.make('BipedalWalker-v3',render_mode="rgb_array") # , render_mode="rgb_array"
    with open(f"data/states_BC.pkl", "rb") as f:
        states = pickle.load(f)
    with open(f"data/actions_BC.pkl", "rb") as f:
        actions = pickle.load(f)
    # BEGIN STUDENT SOLUTION

    if torch.cuda.is_available():
        train_device = eval_device = "cuda"
    else:
        train_device = "mps" if torch.backends.mps.is_available() else "cpu"
        eval_device = "cpu"

    def make_trainer(device):
        model = PolicyDiffusionTransformer(
            num_transformer_layers=6,
            state_dim=24,
            act_dim=4,
            hidden_size=128,
            max_episode_length=1600,
            n_transformer_heads=1,
            device=device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.001)
        return TrainDiffusionPolicy(env, model, optimizer, states, actions, device=device)

    if not skip_training:
        print(f"training on {train_device}")
        make_trainer(train_device).train(
            num_training_steps=num_training_steps,
            batch_size=256,
            wandb_logging=wandb is not None,
            output_dir=output_dir,
        )

    print(f"evaluating on {eval_device}")
    trainer = make_trainer(eval_device)
    eval_results = []

    for num_actions_to_eval_in_a_row in eval_actions_in_a_row:
        eval_results.append(trainer.evaluation(
            num_samples=eval_samples,
            num_actions_to_eval_in_a_row=num_actions_to_eval_in_a_row,
            model_dir=output_dir,
            output_dir=output_dir,
        ))
        with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
            json.dump(eval_results, f, indent=4)
        print(json.dumps(eval_results[-1], indent=4))

    # END STUDENT SOLUTION
    traj_reward = 0
    gif_attempts = 0
    while traj_reward < 240 and (max_gif_attempts is None or gif_attempts < max_gif_attempts):
        rewards, rgbs = trainer.sample_trajectory(trainer.env, num_actions_to_eval_in_a_row=3, render=True)
        traj_reward = rewards.sum()
        gif_attempts += 1
        print(f"got trajectory with reward {traj_reward}")
    imageio.mimsave(os.path.join(output_dir, 'gifs_diffusion.gif'), rgbs, fps=33)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None, help="defaults to the next unused results_diffusion/trial_{number}")
    parser.add_argument('--num_training_steps', type=int, default=50000)
    parser.add_argument('--skip_training', action='store_true', help="evaluate the model already saved in --output_dir")
    parser.add_argument('--eval_samples', type=int, default=20)
    parser.add_argument('--eval_actions_in_a_row', type=int, nargs='+', default=[3, 2, 1])
    parser.add_argument('--max_gif_attempts', type=int, default=None)
    args = parser.parse_args()
    run_training(
        output_dir=args.output_dir,
        num_training_steps=args.num_training_steps,
        skip_training=args.skip_training,
        eval_samples=args.eval_samples,
        eval_actions_in_a_row=args.eval_actions_in_a_row,
        max_gif_attempts=args.max_gif_attempts,
    )
