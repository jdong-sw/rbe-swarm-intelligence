import numpy as np

class SampleCauchyDistribution:
    def __init__(self, rho, sample_size):
        self.rho = rho
        self.sample_size = sample_size
        self.samples = []

        self.x0 = -np.pi
        self.x1 = np.pi

        self.generate_samples()

    def generate_samples(self):
        import time
        t0 = time.time()
        # Generates a list of <size> random samples, obeying the distribution custDist
        # Suggests random samples between x0 and x1 and accepts the suggestion with probability custDist(x)
        #custDist noes not need to be normalized. Add this condition to increase performance.
        #Best performance for max_{x in [x0,x1]} custDist(x) = 1
        print("Generating " + str(self.sample_size) + " samples from a Cauchy distribution with rho = " + str(self.rho) + "...")
        samples = []
        nLoop = 0
        while len(samples) < self.sample_size:
            x = np.random.uniform(low=self.x0, high=self.x1)
            prop = self.cauchy_angle_dist(x)
            assert 0 <= prop <= 1
            if np.random.uniform(low=0, high=1) <= prop:
                samples += [x]
            nLoop += 1
        t1 = time.time()
        total_time = t1-t0
        print("Time spent generating samples: " + str(round(total_time, 2)) + "s.")
        self.samples = samples


    def pick_sample(self):
        return self.samples[np.random.randint(self.sample_size)]

    def cauchy_angle_dist(self, x):
        rho=self.rho
        return (1 - rho**2) / (2*np.pi * (1 + rho**2 - 2*rho*np.cos(x)))

    def check_accuracy(self):
        # Plots the generated samples in a histogram with the PDF overlaid to check the accuracy of the samples.
        import matplotlib.pyplot as plt
        # hist
        bins = np.linspace(self.x0, self.x1, int(self.x1 - self.x0 + 1))
        hist = np.histogram(self.samples, bins)[0]
        hist = hist / np.sum(hist)
        plt.bar((bins[:-1] + bins[1:]) / 2, hist, width=.2, label='Sampled distribution')
        # dist
        grid = np.linspace(self.x0, self.x1, 10000)
        discCustDist = np.array([self.cauchy_angle_dist(x) for x in grid])
        # discCustDist *= 1.0 / (grid[1] - grid[0]) / np.sum(discCustDist)
        plt.plot(grid, discCustDist, label='Cauchy Angle PDF', color='C1', linewidth=4)
        # decoration
        plt.legend(loc=3, bbox_to_anchor=(1, 0))
        plt.show()


# testing
sca = SampleCauchyDistribution(rho=0.05, sample_size=10**5)
sca.check_accuracy()
