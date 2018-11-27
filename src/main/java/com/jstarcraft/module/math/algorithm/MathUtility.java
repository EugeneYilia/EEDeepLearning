package com.jstarcraft.module.math.algorithm;

import com.jstarcraft.module.math.structure.MathIterator;
import com.jstarcraft.module.math.structure.MathScalar;

public class MathUtility {

	public final static float EPSILON = 1E-5F;

	private MathUtility() {
	}

	public static boolean equal(float left, float right) {
		return Math.abs(left - right) < EPSILON;
	}

	/**
	 * Return n!
	 *
	 * @param number
	 *            the given value for n! computation
	 * @return n!
	 */
	public static int factorial(int number) {
		if (number < 0) {
			return 0;
		} else if (number == 0 || number == 1) {
			return 1;
		} else {
			return number * factorial(number - 1);
		}
	}

	public static float logarithm(float number, int base) {
		return (float) (Math.log(number) / Math.log(base));
	}

	/**
	 * Given log(a) and log(b), return log(a + b)
	 *
	 * @param logX
	 *            {@code log(a)}
	 * @param logY
	 *            {@code log(b)}
	 * @return {@code log(a + b)}
	 */
	public static float sumLogarithm(float logX, float logY) {
		float value;
		if (logX < logY) {
			value = (float) (logY + Math.log(1 + Math.exp(logX - logY)));
		} else {
			value = (float) (logX + Math.log(1 + Math.exp(logY - logX)));
		}
		return value;
	}

	/**
	 * logistic function g(x)
	 * 
	 * @param x
	 *            the given parameter x of the function g(x)
	 * @return value of the logistic function g(x)
	 */

	// TODO 似然函数(logistic(x) + logistic(-x) = 1)
	public static float logistic(float x) {
		return (float) (1F / (1F + Math.exp(-x)));
	}

	/**
	 * Return a gaussian value with mean {@code mu} and standard deviation
	 * {@code sigma}.
	 *
	 * @param value
	 *            input value
	 * @param mean
	 *            mean of normal distribution
	 * @param standardDeviation
	 *            standard deviation of normation distribution
	 * @return a gaussian value with mean {@code mu} and standard deviation
	 *         {@code sigma}
	 */
	public static float gaussian(float value, float mean, float standardDeviation) {
		return (float) (Math.exp(-0.5F * Math.pow(value - mean, 2F) / (standardDeviation * standardDeviation)));
	}

	/**
	 * Gradient value of logistic function logistic(x).
	 *
	 * @param x
	 *            parameter x of the function logistic(x)
	 * @return gradient value of logistic function logistic(x)
	 */
	public static float logisticGradientValue(float x) {
		return (float) (logistic(x) * logistic(-x));
	}

	/**
	 * Get the normalized value using min-max normalizaiton.
	 *
	 * @param value
	 *            value to be normalized
	 * @param minimum
	 *            min value
	 * @param maximum
	 *            max value
	 * @return normalized value
	 */
	public static float normalize(float value, float minimum, float maximum) {
		if (maximum > minimum) {
			return (value - minimum) / (maximum - minimum);
		} else if (equal(minimum, maximum)) {
			return value / maximum;
		}
		return value;
	}

	/**
	 * 范数
	 * 
	 * @return
	 */
	public static float norm(float[] values, int power) {
		float norm = 0F;
		for (float value : values) {
			norm += Math.pow(Math.abs(value), power);
		}
		return (float) Math.pow(norm, 1F / power);
	}

	/**
	 * 范数
	 * 
	 * @return
	 */
	public static float norm(MathIterator<?> iterator, int power) {
		float norm = 0F;
		for (MathScalar term : iterator) {
			norm += Math.pow(Math.abs(term.getValue()), power);
		}
		return (float) Math.pow(norm, 1F / power);
	}

	/**
	 * Fabonacci sequence.
	 *
	 * @param number
	 *            length of the sequence
	 * @return sum of the sequence
	 */
	public static int fabonacci(int number) {
		assert number > 0;

		if (number == 1) {
			return 0;
		} else if (number == 2) {
			return 1;
		} else {
			return fabonacci(number - 1) + fabonacci(number - 2);
		}
	}

	/**
	 * Greatest common divisor (gcd) or greatest common factor (gcf)
	 * <p>
	 * reference: http://en.wikipedia.org/wiki/Greatest_common_divisor
	 *
	 * @param x
	 *            given parameter a of the function
	 * @param y
	 *            given parameter b of the function
	 * @return Greatest common divisor (gcd) or greatest common factor (gcf)
	 */
	public static int gcd(int x, int y) {
		if (y == 0) {
			return x;
		} else {
			return gcd(y, x % y);
		}

	}

	/**
	 * least common multiple (lcm).
	 *
	 * @param x
	 *            given parameter a of the function
	 * @param y
	 *            given parameter b of the function
	 * @return least common multiple (lcm)
	 */
	public static int lcm(int x, int y) {
		if (x > 0 && y > 0) {
			return (int) ((0D + x * y) / gcd(x, y));
		} else {
			return 0;
		}

	}

	/**
	 * sqrt(a^2 + b^2) without under/overflow.
	 *
	 * @param x
	 *            given parameter a of the function
	 * @param y
	 *            given parameter b of the function
	 * @return {@code sqrt(a^2 + b^2) without under/overflow}
	 */
	public static float hypot(float x, float y) {
		float value;
		if (Math.abs(x) > Math.abs(y)) {
			value = y / x;
			value = (float) (Math.abs(x) * Math.sqrt(1 + value * value));
		} else if (!equal(y, 0F)) {
			value = x / y;
			value = (float) (Math.abs(y) * Math.sqrt(1 + value * value));
		} else {
			value = 0F;
		}
		return value;
	}

	/**
	 * 伽玛函数
	 * 
	 * @param value
	 * @return
	 */
	public static float gamma(float value) {
		return (float) Math.exp(logGamma(value));
	}

	/**
	 * 伽玛函数的对数
	 * 
	 * @param value
	 * @return
	 */
	public static float logGamma(float value) {
		if (value <= 0F) {
			return Float.NaN;
		}
		float tmp = (float) ((value - 0.5F) * Math.log(value + 4.5F) - (value + 4.5F));
		float ser = 1F + 76.18009173F / (value + 0F) - 86.50532033F / (value + 1F) + 24.01409822F / (value + 2F) - 1.231739516F / (value + 3F) + 0.00120858003F / (value + 4F) - 0.00000536382F / (value + 5F);
		return (float) (tmp + Math.log(ser * Math.sqrt(2F * Math.PI)));
	}

	/**
	 * 伽玛函数的对数的一阶导数
	 * 
	 * @param value
	 * @return
	 */
	public static float digamma(float value) {
		return Digamma.calculate(value);
	}

	/**
	 * 伽玛函数的对数的二阶导数
	 * 
	 * @param value
	 * @return
	 */
	public static float trigamma(float value) {
		return Trigamma.calculate(value);
	}

	public static float inverse(float y, int n) {
		return Digamma.inverse(y, n);
	}

}
