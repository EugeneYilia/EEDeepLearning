package com.jstarcraft.module.neuralnetwork.loss;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * F–measure loss function is a loss function design for training on imbalanced
 * datasets. Essentially, this loss function is a continuous approximation of
 * the F_Beta evaluation measure, of which F_1 is a special case.<br>
 * <br>
 * <b>Note</b>: this implementation differs from that described in the original
 * paper by Pastor-Pellicer et al. in one important way: instead of maximizing
 * the F-measure loss function as presented there (equations 2/3), we minimize
 * 1.0 - FMeasure. Consequently, a score of 0 is the minimum possible value
 * (optimal predictions) and a score of 1.0 is the maximum possible value.<br>
 * <br>
 * This implementation supports 2 types of operation:<br>
 * - Binary: single output/label (Typically sigmoid activation function)<br>
 * - Binary: 2-output/label (softmax activation function + 1-hot labels)<br>
 * Note that the beta value can be configured via the constructor.<br>
 * <br>
 * The following situations are NOT currently supported, may be added in the
 * future:<br>
 * - Multi-label (multiple independent binary outputs)<br>
 * - Multiclass (via micro or macro averaging)<br>
 *
 * <br>
 * Reference: Pastor-Pellicer et al. (2013), F-Measure as the Error Function to
 * Train Neural Networks,
 * <a href="https://link.springer.com/chapter/10.1007/978-3-642-38679-4_37">
 * https://link.springer.com/chapter/10.1007/978-3-642-38679-4_37</a>
 *
 * @author Alex Black
 */
public class FMeasureLossFunction implements LossFunction {

	public static final float DEFAULT_BETA = 1F;

	private final float beta;

	public FMeasureLossFunction() {
		this(DEFAULT_BETA);
	}

	public FMeasureLossFunction(float beta) {
		if (beta <= 0) {
			throw new UnsupportedOperationException("Invalid value: beta must be > 0. Got: " + beta);
		}
		this.beta = beta;
	}

	private KeyValue<Float, Float> computeNumeratorWithDenominator(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		int size = tests.getColumnSize();
		if (size != 1 && size != 2) {
			throw new UnsupportedOperationException("For binary classification: expect output size of 1 or 2. Got: " + size);
		}
		// First: determine positives and negatives
		float tp = 0F;
		float fp = 0F;
		float fn = 0F;

		float isPositiveLabel;
		float isNegativeLabel;
		float pClass0;
		float pClass1;
		if (size == 1) {
			for (int row = 0; row < trains.getRowSize(); row++) {
				if (tests.getValue(row, 0) != 0F) {
					isPositiveLabel = 1F;
					isNegativeLabel = 0F;
				} else {
					isPositiveLabel = 0F;
					isNegativeLabel = 1F;
				}

				float score = trains.getValue(row, 0);
				pClass0 = 1F - score;
				pClass1 = score;

				tp += isPositiveLabel * pClass1;
				fp += isNegativeLabel * pClass1;
				fn += isPositiveLabel * pClass0;
			}
		} else {
			for (int row = 0; row < trains.getRowSize(); row++) {
				isPositiveLabel = tests.getValue(row, 1);
				isNegativeLabel = tests.getValue(row, 0);
				pClass0 = trains.getValue(row, 0);
				pClass1 = trains.getValue(row, 1);

				tp += isPositiveLabel * pClass1;
				fp += isNegativeLabel * pClass1;
				fn += isPositiveLabel * pClass0;
			}
		}

		float numerator = (1F + beta * beta) * tp;
		float denominator = (1F + beta * beta) * tp + beta * beta * fn + fp;

		return new KeyValue<Float, Float>(numerator, denominator);
	}

	@Override
	public float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		KeyValue<Float, Float> keyValue = computeNumeratorWithDenominator(tests, trains, masks);
		float numerator = keyValue.getKey();
		float denominator = keyValue.getValue();
		if (numerator == 0F && denominator == 0F) {
			return 0F;
		}
		return 1F - numerator / denominator;
	}

	@Override
	public void computeGradient(MathMatrix tests, MathMatrix trains, MathMatrix masks, MathMatrix gradients) {
		KeyValue<Float, Float> keyValue = computeNumeratorWithDenominator(tests, trains, masks);
		float numerator = keyValue.getKey();
		float denominator = keyValue.getValue();
		if (numerator == 0F && denominator == 0F) {
			// Zero score -> zero gradient
			gradients.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);
			return;
		}

		float secondTerm = numerator / (denominator * denominator);
		// TODO 避免重复分配内存
		int size = tests.getColumnSize();
		if (size == 1) {
			// Single binary output case
			gradients.mapValues((row, column, value, message) -> {
				value = tests.getValue(row, column) * (1F + beta * beta) / denominator - secondTerm;
				return -value;
			}, null, MathCalculator.PARALLEL);
		} else {
			// Softmax case: the getColumn(1) here is to account for the fact
			// that we're using prob(class1)
			// only in the score function; column(1) is equivalent to output for
			// the single output case
			MathVector vector = gradients.getColumnVector(1);
			MathVector label = tests.getColumnVector(1);
			vector.mapValues((index, value, message) -> {
				value = label.getValue(index) * (1F + beta * beta) / denominator - secondTerm;
				return -value;
			}, null, MathCalculator.SERIAL);
		}

		// Negate relative to description in paper, as we want to *minimize*
		// 1.0-fMeasure, which is equivalent to
		// maximizing fMeasure
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
			FMeasureLossFunction that = (FMeasureLossFunction) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.beta, that.beta);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(beta);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "FMeasureLossFunction(beta=" + beta + ")";
	}

}
