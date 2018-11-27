package com.jstarcraft.module.neuralnetwork.loss;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.ColumnCompositeMatrix;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.DenseVector;

/**
 * This is a cost function associated with a mixture-density network. For
 * background, this is inspired by Bishop's work pioneering the mixture density
 * network. The essence of the idea is that the cost function attempts to model
 * the output as if it were a mixture of gaussian probability densities. The
 * network attempts to converge on a cost function which involves the negative
 * log likelihood of the output fitting a set of data and estimates the "alpha"
 * contribution to of each of the distributions the 'mu' (mean) and 'sigma'
 * (standard deviation) of each of the distributions.
 * 
 * For a full description of the technique, refer to Bishop's work.
 * 
 * Bishop CM. Mixture density networks, Neural Computing Research Group Report:
 * NCRG/94/004, Aston University, Birmingham, 1994
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf
 * 
 * There is no public constructor, please use the builder to create an
 * approriate mixture loss function for the number of outputs and number of
 * mixtures you would like to fit.
 * 
 * Note that this means that the output layer must provide
 * (labelWidth+2)*mixtures output values in order to describe the distributions
 * of vectors of width labelWidth. Please ensure that the size of the output
 * layer matches the number of mixtures provided.
 * 
 * @author Jonathan Arney
 */
public class MixtureDensityLossFunction implements LossFunction {

	private static final double SQRT_TWO_PI = Math.sqrt(2D * Math.PI);

	private int mixtures;

	private int labels;

	/**
	 * This class is a data holder for the mixture density components for convenient
	 * manipulation. These are organized as rank-3 matrices with shape [nSamples,
	 * nLabelsPerSample, nMixturesPerLabel] and refer to the 'alpha' (), 'mu' (),
	 * and 'sigma' ().
	 */
	// weight of that gaussian
	private DenseMatrix weight;
	// mean for that gaussian
	private DenseMatrix[] means;
	// standard-deviation for that gaussian
	private DenseMatrix standardDeviation;

	private DenseMatrix sum, variance, exponent, normal;

	MixtureDensityLossFunction() {
	}

	/**
	 * This method constructs a mixture density cost function which causes the
	 * network to learn a mixture of gaussian distributions for each network output.
	 * The network will learn the 'alpha' (weight for each distribution), the 'mu'
	 * or 'mean' of each distribution, and the 'sigma' (standard-deviation) of the
	 * mixture. Together, this distribution can be sampled according to the
	 * probability density learned by the network.
	 * 
	 * @param mixtures
	 *            Number of gaussian mixtures to model.
	 * @param labelWidth
	 *            Size of the labels vector for each sample.
	 */
	public MixtureDensityLossFunction(int mixtures, int labelWidth) {
		this.mixtures = mixtures;
		this.labels = labelWidth;
		this.means = new DenseMatrix[labelWidth];
	}

	@Override
	public void doCache(MathMatrix tests, MathMatrix trains) {
		int rowSize = tests.getRowSize();
		int columnSize = tests.getColumnSize();
		if (columnSize != labels * mixtures) {
			throw new IllegalArgumentException("tests size " + columnSize + " must be labels*mixtures where labels = " + labels + " and mixtures = " + mixtures);
		}
		weight = DenseMatrix.valueOf(rowSize, mixtures);
		for (int index = 0, size = labels; index < size; index++) {
			means[index] = DenseMatrix.valueOf(rowSize, mixtures);
		}
		standardDeviation = DenseMatrix.valueOf(rowSize, mixtures);
		sum = DenseMatrix.valueOf(rowSize, mixtures);
		variance = DenseMatrix.valueOf(rowSize, mixtures);
		exponent = DenseMatrix.valueOf(rowSize, mixtures);
		normal = DenseMatrix.valueOf(rowSize, mixtures);
	}

	// This method extracts the "alpha", "mu", and "sigma" from the
	// output of the neural network.
	// This is done manually, but it should ultimately be done
	// through Nd4j operations in order to increase performance.
	private void extractComponents(MathMatrix trains) {
		int rowSize = trains.getRowSize();
		int columnSize = trains.getColumnSize();
		if (columnSize != (labels + 2) * mixtures) {
			throw new IllegalArgumentException("trains size " + columnSize + " must be (labels+2)*mixtures where labels = " + labels + " and mixtures = " + mixtures);
		}

		// Output is 2 dimensional (samples, labels)
		//
		// For each label vector of length 'labels', we will have
		// an output vector of length '(labels + 2) * nMixtures.
		// The first nMixtures outputs will correspond to the 'alpha' for each
		// mixture.
		// The second nMixtures outputs will correspond to the 'sigma' and the
		// last nMixtures*labels
		// will correspond to the 'mu' (mean) of the output.

		// Reorganize these.
		// alpha = samples, 0 to nMixtures
		// mu = samples, nMixtures to 2*nMixtures
		// sigma = samples, 2*nMixtures to (labels + 2)*nMixtures
		// Alpha is then sub-divided through reshape by mixtures per label and
		// samples.

		weight.mapValues(MatrixMapper.copyOf(ColumnCompositeMatrix.detachOf(ColumnCompositeMatrix.class.cast(trains), 0, mixtures)), null, MathCalculator.PARALLEL);
		for (int columnIndex = 0; columnIndex < weight.getColumnSize(); columnIndex++) {
			float maximum = Float.NEGATIVE_INFINITY;
			MathVector vector = weight.getColumnVector(columnIndex);
			for (int rowIndex = 0; rowIndex < weight.getRowSize(); rowIndex++) {
				float value = vector.getValue(rowIndex);
				if (maximum < value) {
					maximum = value;
				}
			}
			float sum = 0F;
			for (int rowIndex = 0; rowIndex < weight.getRowSize(); rowIndex++) {
				float value = vector.getValue(rowIndex);
				value = (float) Math.exp(value - maximum);
				vector.setValue(rowIndex, value);
				sum += value;
			}
			for (int rowIndex = 0; rowIndex < weight.getRowSize(); rowIndex++) {
				float value = vector.getValue(rowIndex);
				vector.setValue(rowIndex, value / sum);
			}
		}

		standardDeviation.mapValues(MatrixMapper.copyOf(ColumnCompositeMatrix.detachOf(ColumnCompositeMatrix.class.cast(trains), mixtures, 2 * mixtures)), null, MathCalculator.PARALLEL);
		for (int index = 0, size = means.length; index < size; index++) {
			means[index].mapValues(MatrixMapper.copyOf(ColumnCompositeMatrix.detachOf(ColumnCompositeMatrix.class.cast(trains), (2 + index) * mixtures, (3 + index) * mixtures)), null, MathCalculator.PARALLEL);
		}

		// Alpha is a softmax because
		// the alpha should all sum to 1 for a given gaussian mixture.
		// TODO
		// alpha = Nd4j.getExecutioner().execAndReturn(new SoftMax(alpha));

		// Mu comes directly from the network as an unmolested value.
		// Note that this effectively means that the output layer of
		// the network should have an activation function at least as large as
		// the expected values. It is best for the output
		// layer to be an IDENTITY activation function.
		// mdc.mu = mdc.mu;

		// Sigma comes from the network as an exponential in order to
		// ensure that it is positive-definite.
		standardDeviation.mapValues((row, column, value, message) -> {
			return (float) FastMath.exp(value);
		}, null, MathCalculator.PARALLEL);
	}

	private void labelsMinusMu(MathMatrix tests) {
		// Now that we have the mixtures, let's compute the negative
		// log likelihodd of the label against the

		// This worked, but was actually much
		// slower than the for loop below.
		// labels = samples, mixtures, labels
		// mu = samples, mixtures
		// INDArray labelMinusMu = labels
		// .reshape('f', nSamples, labelsPerSample, 1)
		// .repeat(2, mMixtures)
		// .permute(0, 2, 1)
		// .subi(mu);

		// The above code does the same thing as the loop below,
		// but it does it with index magix instead of a for loop.
		// It turned out to be way less efficient than the simple 'for' here.
		for (int index = 0, size = means.length; index < size; index++) {
			MathMatrix matrix = ColumnCompositeMatrix.detachOf(ColumnCompositeMatrix.class.cast(tests), index * mixtures, (index + 1) * mixtures);
			means[index].mapValues((row, column, value, message) -> {
				value = matrix.getValue(row, column) - value;
				return value;
			}, null, MathCalculator.PARALLEL);
		}
	}

	@Override
	public float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		extractComponents(trains);
		labelsMinusMu(tests);
		sum.mapValues((row, column, value, message) -> {
			value = 0F;
			for (int index = 0, size = means.length; index < size; index++) {
				value += (float) FastMath.pow(means[index].getValue(row, column), 2F);
			}
			return value;
		}, null, MathCalculator.PARALLEL);

		float score = 0F;
		for (int rowIndex = 0, rowSize = standardDeviation.getRowSize(); rowIndex < rowSize; rowIndex++) {
			// See Bishop equation (28,29)
			float likelihood = 0F;
			for (int columnIndex = 0, columnSize = standardDeviation.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				// probability density function.
				// See Bishop equation (23)
				// 1/sqrt(2PIs^2) * e^((in-u)^2/2*s^2)
				float value = standardDeviation.getValue(rowIndex, columnIndex);
				value = -(value * value * 2);
				value = (float) FastMath.exp(sum.getValue(rowIndex, columnIndex) / value);
				value /= (float) FastMath.pow(standardDeviation.getValue(rowIndex, columnIndex) * SQRT_TWO_PI, labels);
				value *= weight.getValue(rowIndex, columnIndex);
				likelihood += value;
			}
			score -= (float) FastMath.log(likelihood);
		}
		// TODO 暂时不处理masks
		return score;
	}

	@Override
	public void computeGradient(MathMatrix tests, MathMatrix trains, MathMatrix masks, MathMatrix gradients) {
		extractComponents(trains);
		labelsMinusMu(tests);
		sum.mapValues((row, column, value, message) -> {
			value = 0F;
			for (int index = 0, size = means.length; index < size; index++) {
				value += (float) FastMath.pow(means[index].getValue(row, column), 2D);
			}
			return value;
		}, null, MathCalculator.PARALLEL);

		// This computes pi_i, see Bishop equation (30).
		// See
		// http://www.plsyard.com/dealing-overflow-and-underflow-in-softmax-function/
		// this post for why we calculate the pi_i in this way.
		// With the exponential function here, we have to be very careful
		// about overflow/underflow considerations even with
		// fairly intermediate values. Subtracting the max
		// here helps to ensure over/underflow does not happen here.
		// This isn't exactly a softmax because there's an 'alpha' coefficient
		// here, but the technique works, nonetheless.
		variance.mapValues((row, column, value, message) -> {
			value = standardDeviation.getValue(row, column);
			return value * value;
		}, null, MathCalculator.PARALLEL);
		exponent.mapValues((row, column, value, message) -> {
			value = sum.getValue(row, column);
			return value / -(variance.getValue(row, column) * 2);
		}, null, MathCalculator.PARALLEL);
		for (int rowIndex = 0, rowSize = exponent.getRowSize(); rowIndex < rowSize; rowIndex++) {
			DenseVector vector = exponent.getRowVector(rowIndex);
			float maximum = Float.NEGATIVE_INFINITY;
			for (int columnIndex = 0, columnSize = exponent.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				float value = vector.getValue(columnIndex);
				if (maximum < value) {
					maximum = value;
				}
			}
			for (int columnIndex = 0, columnSize = exponent.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				float value = vector.getValue(columnIndex);
				vector.setValue(columnIndex, value - maximum);
			}
		}
		normal.mapValues((row, column, value, message) -> {
			value = weight.getValue(row, column);
			value /= (float) FastMath.pow(standardDeviation.getValue(row, column) * SQRT_TWO_PI, labels) * FastMath.exp(exponent.getValue(row, column));
			return value;
		}, null, MathCalculator.PARALLEL);
		for (int rowIndex = 0, rowSize = normal.getRowSize(); rowIndex < rowSize; rowIndex++) {
			DenseVector vector = normal.getRowVector(rowIndex);
			float sum = 0F;
			for (int columnIndex = 0, columnSize = normal.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				float value = vector.getValue(columnIndex);
				sum += value;
			}
			for (int columnIndex = 0, columnSize = normal.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				float value = vector.getValue(columnIndex);
				vector.setValue(columnIndex, value / sum);
			}
		}

		// See Bishop equation (35)
		MathMatrix weightGradient = ColumnCompositeMatrix.detachOf(ColumnCompositeMatrix.class.cast(gradients), 0, mixtures);
		weightGradient.mapValues((row, column, value, message) -> {
			return weight.getValue(row, column) - normal.getValue(row, column);
		}, null, MathCalculator.PARALLEL);
		// See Bishop equation (38)
		MathMatrix sigmaGradient = ColumnCompositeMatrix.detachOf(ColumnCompositeMatrix.class.cast(gradients), mixtures, 2 * mixtures);
		sigmaGradient.mapValues((row, column, value, message) -> {
			return -(sum.getValue(row, column) / variance.getValue(row, column) - labels) * normal.getValue(row, column);
		}, null, MathCalculator.PARALLEL);
		// See Bishop equation (39)
		for (int index = 0; index < labels; index++) {
			DenseMatrix mean = means[index];
			MathMatrix meanGradient = ColumnCompositeMatrix.detachOf(ColumnCompositeMatrix.class.cast(gradients), (2 + index) * mixtures, (3 + index) * mixtures);
			meanGradient.mapValues((row, column, value, message) -> {
				return -(mean.getValue(row, column) * normal.getValue(row, column) / variance.getValue(row, column));
			}, null, MathCalculator.PARALLEL);
		}
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
			MixtureDensityLossFunction that = (MixtureDensityLossFunction) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.mixtures, that.mixtures);
			equal.append(this.labels, that.labels);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(mixtures);
		hash.append(labels);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "MixtureDensityLossFunction(mixtures=" + mixtures + ", labels=" + labels + ")";
	}

}
