'''
flipflop.py
Written for Python 3.8.17
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import os
import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

class FlipFlopData(object):

	def __init__(self, 
		n_bits=3,
		n_time=64,
		p=0.5, 
		random_seed=None):
		''' Creates a FlipFlopData object.

		Args:
			n_bits: Non-negative integer. Specifies the number of channels in
			the flip flop memory. Default: 3

			n_time: Non-negative integer. Specifies the number of timesteps per
			trial. Default: 64.

			p: float in [0, 1]. Specifies the probability of a non-zero input
			pulse at each timestep. Non-zero inputs are set to either +1 or -1,
			with equal probability.

		Returns:
			None.

		'''

		self.rng = npr.RandomState(random_seed)
		self.n_time = n_time
		self.n_bits = n_bits
		self.p = p

	def generate_data(self, n_trials, dtype=np.float32):
		''' Generates trial data for training, testing, or validating a FlipFlop
		memory device.

		Args:
			n_trials: Non-negative integer. Specifies the number of trials to 
			generate.

			dtype: Numpy datatype for the generated data. Default: np.float32.

		Returns:
			Dict containing keys:
				'inputs': A (n_trials, n_time, n_bits) Numpy array containing
				the input sequences.

				'targets': A (n_trials, n_time, n_bits) Numpy array containing
				the target output sequences.

				A function, F, that perfectly solves the flip flop memory task 
				will satisfy: F(inputs[i]) = targets[i], for all trials i.
		'''

		n_time = self.n_time
		n_bits = self.n_bits
		p = self.p

		# Randomly generate unsigned input pulses
		unsigned_inputs = self.rng.binomial(
			1, p, [n_trials, n_time, n_bits])

		# Ensure every trial is initialized with a pulse at time 0
		unsigned_inputs[:, 0, :] = 1

		# Generate random signs {-1, +1}
		random_signs = 2*self.rng.binomial(
			1, 0.5, [n_trials, n_time, n_bits]) - 1

		# Apply random signs to input pulses
		inputs = np.multiply(unsigned_inputs, random_signs)

		# Allocate targets
		targets = np.zeros([n_trials, n_time, n_bits])

		# Update inputs (zero-out random start holds) & compute targets
		for trial_idx in range(n_trials):
			for bit_idx in range(n_bits):
				input_ = np.squeeze(inputs[trial_idx, :, bit_idx])
				t_flip = np.where(input_ != 0)
				for flip_idx in range(np.size(t_flip)):
					# Get the time of the next flip
					t_flip_i = t_flip[0][flip_idx]

					'''Set the targets to the sign of the flip for the
					remainder of the trial. Future flips will overwrite future
					targets'''
					targets[trial_idx, t_flip_i:, bit_idx] = \
						inputs[trial_idx, t_flip_i, bit_idx]

		return {
			'inputs': inputs.astype(np.float32),
			'targets': targets.astype(np.float32)
			}

	@classmethod
	def plot_trials(cls, data, pred, start_time=0, stop_time=None, 
		n_trials_plot=1,
		fig=None):
		'''Plots example trials, complete with input pulses, correct target
		outputs, and RNN-predicted outputs.

		Args:
			data: dict as returned by generate_data.

			start_time (optional): int specifying the first timestep to plot.
			Default: 0.

			stop_time (optional): int specifying the last timestep to plot.
			Default: n_time.

		Returns:
			None.
		'''

		FIG_WIDTH = 6 # inches
		FIG_HEIGHT = 3*n_trials_plot # inches

		if fig is None:
			fig = plt.figure(
				figsize=(FIG_WIDTH, FIG_HEIGHT),
				tight_layout=True)
		else:
			fig.clf()

		inputs = data['inputs']
		targets = data['targets']
		pred_output = pred['output']

		n_batch, n_time, _ = inputs.shape
		n_plot = np.min([n_trials_plot, n_batch])

		if stop_time is None:
			stop_time = n_time

		time_idx = list(range(start_time, stop_time))

		for trial_idx in range(n_plot):
			ax = plt.subplot(n_plot, 1, trial_idx+1)
			if n_plot == 1:
				plt.title('Example trial', fontweight='bold')
			else:
				plt.title('Example trial %d' % (trial_idx + 1),
						  fontweight='bold')

			cls._plot_single_trial(
				inputs[trial_idx, time_idx, :],
				targets[trial_idx, time_idx, :],
				pred_output[trial_idx, time_idx, :])

			# Only plot x-axis ticks and labels on the bottom subplot
			if trial_idx < (n_plot-1):
				plt.xticks([])
			else:
				plt.xlabel('Timestep', fontweight='bold')

		cls._refresh_figs()

		return fig

	@classmethod
	def _plot_single_trial(cls, input_txd, output_txd, pred_output_txd):
		''' Plots a single example trial, complete with input pulses, correct 
		target outputs, and RNN-predicted outputs.
		'''

		VERTICAL_SPACING = 2.5
		[n_time, n_bits] = input_txd.shape
		tt = list(range(n_time))

		y_ticks = [VERTICAL_SPACING*bit_idx for bit_idx in range(n_bits)]
		y_tick_labels = \
			['Bit %d' % (n_bits-bit_idx) for bit_idx in range(n_bits)]

		plt.yticks(y_ticks, y_tick_labels, fontweight='bold')
		for bit_idx in range(n_bits):

			vertical_offset = VERTICAL_SPACING*bit_idx

			# Input pulses
			plt.fill_between(
				tt,
				vertical_offset + input_txd[:, bit_idx],
				vertical_offset,
				step='mid',
				color='gray')

			# Correct outputs
			plt.step(
				tt,
				vertical_offset + output_txd[:, bit_idx],
				where='mid',
				linewidth=2,
				color='cyan')

			# RNN outputs
			plt.step(
				tt,
				vertical_offset + pred_output_txd[:, bit_idx],
				where='mid',
				color='purple',
				linewidth=1.5,
				linestyle='--')

		plt.xlim(-1, n_time)

	@classmethod
	def _refresh_figs(cls):
		''' Refreshes all matplotlib figures.

		Args:
			None.

		Returns:
			None.
		'''
		plt.ion()
		plt.show()
		plt.pause(1e-10)


if __name__ == '__main__':
	# Example usage
	data_generator = FlipFlopData(n_bits=1, n_time=64, p=0.5)
	data = data_generator.generate_data(n_trials=1)
	print(type(data['inputs']))

	print(data['inputs'][0])
	data_generator._plot_single_trial(data['inputs'][0], data['targets'][0], data['targets'][0])
	plt.legend(['Input', 'Target', 'RNN output'])
	plt.show()	