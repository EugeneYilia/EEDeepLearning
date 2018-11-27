package com.jstarcraft.module.neuralnetwork.loss;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;

// multinomial
/**
 *
 * Multi-Class Cross Entropy loss function:<br>
 * L = sum_i actual_i * log( predicted_i )
 *
 * @author Alex Black, Susan Eraly
 * @see NegativeLogLikelihoodLossFunction
 */
public class MCXENTLossFunction implements LossFunction {

	private static final float DEFAULT_CLIP = 1E-10F;

	private float clip;

	private boolean isSoftMaximum;

	MCXENTLossFunction() {
	}

	/**
	 * Multi-Class Cross Entropy loss function where each the output is (optionally)
	 * weighted/scaled by a flags scalar value. Note that the weights array must be
	 * a row vector, of length equal to the labels/output dimension 1 size. A weight
	 * vector of 1s should give identical results to no weight vector.
	 *
	 * @param weights
	 *            Weights array (row vector). May be null.
	 */
	public MCXENTLossFunction(boolean isSoftMaximum) {
		this(DEFAULT_CLIP, isSoftMaximum);
	}

	/**
	 * Multi-Class Cross Entropy loss function where each the output is (optionally)
	 * weighted/scaled by a fixed scalar value. Note that the weights array must be
	 * a row vector, of length equal to the labels/output dimension 1 size. A weight
	 * vector of 1s should give identical results to no weight vector.
	 *
	 * @param weights
	 *            Weights array (row vector). May be null.
	 */
	public MCXENTLossFunction(float clip, boolean isSoftMaximum) {
		if (clip < 0 || clip > 0.5) {
			throw new IllegalArgumentException("Invalid clipping epsilon: epsilon should be >= 0 (but near zero). Got: " + clip);
		}
		this.clip = clip;
		this.isSoftMaximum = isSoftMaximum;
	}

	@Override
	public float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		float score = 0F;
		if (isSoftMaximum && clip > 0F) {
			float minimum = clip;
			float maximum = 1F - clip;
			for (MatrixScalar term : trains) {
				float value = term.getValue();
				value = value < minimum ? minimum : (value > maximum ? maximum : value);
				value = (float) (FastMath.log(value) * tests.getValue(term.getRow(), term.getColumn()));
				score += -value;
			}
		} else {
			for (MatrixScalar term : trains) {
				float value = term.getValue();
				value = (float) (FastMath.log(value) * tests.getValue(term.getRow(), term.getColumn()));
				score += -value;
			}
		}
		// TODO 暂时不处理masks
		return score;
	}

	@Override
	public void computeGradient(MathMatrix tests, MathMatrix trains, MathMatrix masks, MathMatrix gradients) {
		// TODO 此处用了技巧,将activation与loss的反向传播合并.
		// if (isSoftMaximum) {
		// gradients.calculate((row, column, value) -> {
		// value = trains.getTermValue(row, column) - tests.getTermValue(row,
		// column);
		// return value;
		// });
		// } else {
		// gradients.calculate((row, column, value) -> {
		// value = trains.getTermValue(row, column);
		// value = -(tests.getTermValue(row, column) / value);
		// return value;
		// });
		// }
		gradients.mapValues((row, column, value, message) -> {
			value = trains.getValue(row, column);
			value = -(tests.getValue(row, column) / value);
			return value;
		}, null, MathCalculator.PARALLEL);
		// TODO 暂时不处理masks
	}

	@Override
	public boolean equals(Object object) {
		if (this == object) {
			return true;
		}
		if (object == null) {
			return false;
		}
		if (getClass() != object.getClass()) {
			return false;
		} else {
			MCXENTLossFunction that = (MCXENTLossFunction) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.clip, that.clip);
			equal.append(this.isSoftMaximum, that.isSoftMaximum);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(clip);
		hash.append(isSoftMaximum);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "MCXENTLossFunction(clip=" + clip + ", isSoftMaximum=" + isSoftMaximum + ")";
	}

}
