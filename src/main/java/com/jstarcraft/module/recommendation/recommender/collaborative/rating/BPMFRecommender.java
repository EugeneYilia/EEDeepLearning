package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.distribution.ContinuousProbability;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MatrixUtility;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.exception.RecommendationException;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Salakhutdinov and Mnih, <strong>Bayesian Probabilistic Matrix Factorization
 * using Markov Chain Monte Carlo</strong>, ICML 2008.
 * <p>
 * Matlab version is provided by the authors via
 * <a href="http://www.utstat.toronto.edu/~rsalakhu/BPMF.html">this link</a>.
 * This implementation is modified from the BayesianPMF by the PREA package.
 * Bayesian Probabilistic Matrix Factorization
 */
public class BPMFRecommender extends MatrixFactorizationRecommender {

	private float userMean, userWishart;

	private float itemMean, itemWishart;

	private float userBeta, itemBeta;

	private float rateSigma;

	private int gibbsIterations;

	private DenseMatrix[] userMatrixes;

	private DenseMatrix[] itemMatrixes;

	private ContinuousProbability normalDistribution;
	private ContinuousProbability[] userGammaDistributions;
	private ContinuousProbability[] itemGammaDistributions;

	private class HyperParameter {

		// 缓存
		private float[] thisVectorCache;
		private float[] thatVectorCache;
		private float[] thisMatrixCache;
		private float[] thatMatrixCache;

		private DenseVector factorMeans;

		private DenseMatrix factorVariances;

		private DenseVector randoms;

		private DenseVector outerMeans, innerMeans;

		private DenseMatrix covariance, cholesky, inverse, transpose, gaussian, gamma, wishart, copy;

		HyperParameter(int cache, DenseMatrix factors) {
			if (cache < numberOfFactors) {
				cache = numberOfFactors;
			}
			thisVectorCache = new float[cache];
			thisMatrixCache = new float[cache * numberOfFactors];
			thatVectorCache = new float[cache];
			thatMatrixCache = new float[cache * numberOfFactors];

			factorMeans = DenseVector.valueOf(numberOfFactors);
			float scale = factors.getRowSize();
			for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
				factorMeans.setValue(factorIndex, factors.getColumnVector(factorIndex).getSum(false) / scale);
			}
			outerMeans = DenseVector.valueOf(factors.getRowSize());
			innerMeans = DenseVector.valueOf(factors.getRowSize());
			covariance = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);

			cholesky = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);

			inverse = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
			transpose = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);

			randoms = DenseVector.valueOf(numberOfFactors);
			gaussian = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
			gamma = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
			wishart = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);

			copy = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);

			factorVariances = MatrixUtility.inverse(MatrixUtility.covariance(factors, outerMeans, innerMeans, covariance), copy, inverse);
		}

		/**
		 * 取样
		 * 
		 * @param hyperParameter
		 * @param factors
		 * @param normalMu
		 * @param normalBeta
		 * @param wishartScale
		 * @return
		 * @throws RecommendationException
		 */
		private void sampleParameter(ContinuousProbability[] gammaDistributions, DenseMatrix factors, float normalMu, float normalBeta, float wishartScale) throws RecommendationException {
			int rowSize = factors.getRowSize();
			int columnSize = factors.getColumnSize();
			// 重复利用内存.
			DenseVector meanCache = DenseVector.valueOf(numberOfFactors, thisVectorCache);
			float scale = factors.getRowSize();
			meanCache.mapValues((index, value, message) -> {
				return factors.getColumnVector(index).getSum(false) / scale;
			}, null, MathCalculator.SERIAL);
			float beta = normalBeta + rowSize;
			DenseMatrix populationVariance = MatrixUtility.covariance(factors, outerMeans, innerMeans, covariance);
			DenseMatrix wishartMatrix = wishart.mapValues((row, column, value, message) -> {
				value = 0F;
				if (row == column) {
					value = wishartScale;
				}
				value += populationVariance.getValue(row, column) * rowSize;
				value += (normalMu - meanCache.getValue(row)) * (normalMu - meanCache.getValue(column)) * (normalBeta * rowSize / beta);
				return value;
			}, null, MathCalculator.PARALLEL);
			wishartMatrix = MatrixUtility.inverse(wishartMatrix, copy, inverse);
			wishartMatrix.addMatrix(transpose.copyMatrix(wishartMatrix, true), false).scaleValues(0.5F);
			wishartMatrix = MatrixUtility.wishart(wishartMatrix, normalDistribution, gammaDistributions, randoms, cholesky, gaussian, gamma, transpose, wishart);
			if (wishartMatrix != null) {
				factorVariances = wishartMatrix;
			}
			DenseMatrix normalVariance = MatrixUtility.cholesky(MatrixUtility.inverse(factorVariances, copy, inverse).scaleValues(normalBeta), cholesky);
			if (normalVariance != null) {
				randoms.mapValues(VectorMapper.distributionOf(normalDistribution), null, MathCalculator.SERIAL);
				factorMeans.dotProduct(normalVariance, false, randoms, MathCalculator.SERIAL).mapValues((index, value, message) -> {
					return value + (normalMu * normalBeta + meanCache.getValue(index) * rowSize) * (1F / beta);
				}, null, MathCalculator.SERIAL);
			}
		}

		/**
		 * 更新
		 * 
		 * @param factorMatrix
		 * @param scoreVector
		 * @param hyperParameter
		 * @return
		 * @throws RecommendationException
		 */
		private void updateParameter(DenseMatrix factorMatrix, SparseVector scoreVector, DenseVector factorVector) throws RecommendationException {
			int size = scoreVector.getElementSize();
			// 重复利用内存.
			DenseMatrix factorCache = DenseMatrix.valueOf(size, numberOfFactors, thisMatrixCache);
			MathVector meanCache = DenseVector.valueOf(size, thisVectorCache);
			int index = 0;
			for (VectorScalar term : scoreVector) {
				meanCache.setValue(index, term.getValue() - meanOfScore);
				factorCache.getRowVector(index).mapValues(VectorMapper.copyOf(factorMatrix.getRowVector(term.getIndex())), null, MathCalculator.SERIAL);
				index++;
			}
			transpose.dotProduct(factorCache, true, factorCache, false, MathCalculator.SERIAL);
			DenseMatrix covariance = transpose.mapValues((row, column, value, message) -> {
				return value * rateSigma + factorVariances.getValue(row, column);
			}, null, MathCalculator.SERIAL);
			covariance = MatrixUtility.inverse(covariance, copy, inverse);
			// 重复利用内存.
			meanCache = DenseVector.valueOf(factorCache.getColumnSize(), thatVectorCache).dotProduct(factorCache, true, meanCache, MathCalculator.SERIAL);
			meanCache.scaleValues(rateSigma);
			// 重复利用内存.
			meanCache.addVector(DenseVector.valueOf(factorVariances.getRowSize(), thisVectorCache).dotProduct(factorVariances, false, factorMeans, MathCalculator.SERIAL));
			// 重复利用内存.
			meanCache = DenseVector.valueOf(covariance.getRowSize(), thisVectorCache).dotProduct(covariance, false, meanCache, MathCalculator.SERIAL);
			covariance = MatrixUtility.cholesky(covariance, cholesky);
			if (covariance != null) {
				randoms.mapValues(VectorMapper.distributionOf(normalDistribution), null, MathCalculator.SERIAL);
				factorVector.dotProduct(covariance, false, randoms, MathCalculator.SERIAL).addVector(meanCache);
			} else {
				factorVector.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
			}
		}
	}

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		userMean = configuration.getFloat("rec.recommender.user.mu", 0F);
		userBeta = configuration.getFloat("rec.recommender.user.beta", 1F);
		userWishart = configuration.getFloat("rec.recommender.user.wishart.scale", 1F);

		itemMean = configuration.getFloat("rec.recommender.item.mu", 0F);
		itemBeta = configuration.getFloat("rec.recommender.item.beta", 1F);
		itemWishart = configuration.getFloat("rec.recommender.item.wishart.scale", 1F);

		rateSigma = configuration.getFloat("rec.recommender.rating.sigma", 2F);

		gibbsIterations = configuration.getInteger("rec.recommender.iterations.gibbs", 1);

		userMatrixes = new DenseMatrix[numberOfEpoches - 1];
		itemMatrixes = new DenseMatrix[numberOfEpoches - 1];

		normalDistribution = new ContinuousProbability(new NormalDistribution(new JDKRandomGenerator(numberOfFactors), 0D, 1D));
		userGammaDistributions = new ContinuousProbability[numberOfFactors];
		itemGammaDistributions = new ContinuousProbability[numberOfFactors];
		for (int index = 0; index < numberOfFactors; index++) {
			RandomGenerator random = new JDKRandomGenerator(index);
			userGammaDistributions[index] = new ContinuousProbability(new GammaDistribution(random, (numberOfUsers + numberOfFactors - (index + 1D)) / 2D, 2D));
			itemGammaDistributions[index] = new ContinuousProbability(new GammaDistribution(random, (numberOfItems + numberOfFactors - (index + 1D)) / 2D, 2D));
		}
	}

	@Override
	protected void doPractice() {
		int cacheSize = 0;
		SparseVector[] userVectors = new SparseVector[numberOfUsers];
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector userVector = trainMatrix.getRowVector(userIndex);
			cacheSize = cacheSize < userVector.getElementSize() ? userVector.getElementSize() : cacheSize;
			userVectors[userIndex] = userVector;
		}

		SparseVector[] itemVectors = new SparseVector[numberOfItems];
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
			cacheSize = cacheSize < itemVector.getElementSize() ? itemVector.getElementSize() : cacheSize;
			itemVectors[itemIndex] = itemVector;
		}

		// TODO 此处考虑重构
		HyperParameter userParameter = new HyperParameter(cacheSize, userFactors);
		HyperParameter itemParameter = new HyperParameter(cacheSize, itemFactors);
		for (int iterationStep = 0; iterationStep < numberOfEpoches; iterationStep++) {
			userParameter.sampleParameter(userGammaDistributions, userFactors, userMean, userBeta, userWishart);
			itemParameter.sampleParameter(itemGammaDistributions, itemFactors, itemMean, itemBeta, itemWishart);
			for (int gibbsIteration = 0; gibbsIteration < gibbsIterations; gibbsIteration++) {
				for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
					SparseVector scoreVector = userVectors[userIndex];
					if (scoreVector.getElementSize() == 0) {
						continue;
					}
					userParameter.updateParameter(itemFactors, scoreVector, userFactors.getRowVector(userIndex));
				}
				for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
					SparseVector scoreVector = itemVectors[itemIndex];
					if (scoreVector.getElementSize() == 0) {
						continue;
					}
					itemParameter.updateParameter(userFactors, scoreVector, itemFactors.getRowVector(itemIndex));
				}
			}

			if (iterationStep > 0) {
				userMatrixes[iterationStep - 1] = DenseMatrix.copyOf(userFactors, (row, column, value, message) -> {
					return value;
				});
				itemMatrixes[iterationStep - 1] = DenseMatrix.copyOf(itemFactors, (row, column, value, message) -> {
					return value;
				});
			}
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = 0F;
		for (int iterationStep = 0; iterationStep < numberOfEpoches - 1; iterationStep++) {
			DenseVector userVector = userMatrixes[iterationStep].getRowVector(userIndex);
			DenseVector itemVector = itemMatrixes[iterationStep].getRowVector(itemIndex);
			value = (value * (iterationStep) + meanOfScore + scalar.dotProduct(userVector, itemVector).getValue()) / (iterationStep + 1);
		}
		return value;
	}

}
