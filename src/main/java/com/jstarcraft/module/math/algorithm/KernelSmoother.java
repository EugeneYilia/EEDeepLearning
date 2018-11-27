package com.jstarcraft.module.math.algorithm;

/**
 * This is a class implementing kernel smoothing functions used in Local
 * Low-Rank Matrix Approximation (LLORMA).
 *
 * @author Joonseok Lee
 * @version 1.2
 * @since 2013. 6. 11
 */
public enum KernelSmoother {

	TRIANGULAR_KERNEL, UNIFORM_KERNEL, EPANECHNIKOV_KERNEL, GAUSSIAN_KERNEL;

	public float kernelize(float similarity, float width) {
		float distance = 1F - similarity;
		switch (this) {
		case TRIANGULAR_KERNEL:
			return Math.max(1F - distance / width, 0F);
		case UNIFORM_KERNEL:
			return distance < width ? 1F : 0F;
		case EPANECHNIKOV_KERNEL:
			return (float) Math.max(3F / 4F * (1F - Math.pow(distance / width, 2F)), 0F);
		case GAUSSIAN_KERNEL:
			return (float) (1F / Math.sqrt(2F * Math.PI) * Math.exp(-0.5F * Math.pow(distance / width, 2F)));
		default:
			return Math.max(1F - distance / width, 0F);
		}
	}

}
