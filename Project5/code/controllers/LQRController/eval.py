import matplotlib.pyplot as plt
import numpy as np

def evaluateLQR():
	""" Draw plots based on saved data and calculate score.

	Note:
		You should have r_hist_ex1.npy (the reference trajectory)
						x_hist_ex1.npy (lqr_trajectory)
		in the current directory

		The script will display plot and output score in the terminal in webots

	"""
	r_hist = np.load("r_hist_ex1.npy")
	r_hist = np.array(r_hist)

	x_hist_nf = np.load("x_hist_ex1.npy")
	x_hist_nf = np.array(x_hist_nf)

	lowerBar = 0.5
	upperBar = 0.8

	metric = 2
	ref = sum(abs(r_hist[:, metric]))
	diff_studentResult = sum(abs(r_hist[:, metric] - x_hist_nf[:, metric]))

	error = diff_studentResult/ref
	if error <= lowerBar:
		score = 50  
	elif error >= upperBar:
		score = 0
	else:
		score = 50 - (error - lowerBar)/(upperBar - lowerBar) * 50

	print("="*15 + "YOUR RESULT" + "="*15)
	print("ERROR: {:.3f}".format(error))
	print("SCORE: {:.3f}".format(score))

	# plot
	s = 2
	plt.figure(1)
	plt.plot(r_hist[:,s], 'k', label="Command")
	plt.plot(x_hist_nf[:,s], label="LQR")
	plt.xlabel("s (seconds)")
	plt.ylabel("m (height)")
	plt.legend(loc="upper right")
	plt.show()
