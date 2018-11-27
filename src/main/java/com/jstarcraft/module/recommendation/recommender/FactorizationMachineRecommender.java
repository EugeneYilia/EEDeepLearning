package com.jstarcraft.module.recommendation.recommender;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.distribution.ContinuousProbability;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.tensor.SparseTensor;
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.exception.RecommendationException;

/**
 * Factorization Machine Recommender
 *
 * Rendle, Steffen, et al., <strong>Fast Context-aware Recommendations with
 * Factorization Machines</strong>, SIGIR, 2011.
 *
 * @author Tang Jiaxi and Ma Chen
 */
// TODO 论文中需要支持组合特征(比如:历史评价过的电影),现在的代码并没有实现.
public abstract class FactorizationMachineRecommender extends ModelRecommender {

	/** 训练矩阵(TODO 准备改名为actionTensor或者scoreTensor) */
	protected SparseTensor trainTensor;

	/** 测试矩阵(TODO 准备取消) */
	protected SparseTensor testTensor;

	/**
	 * global bias
	 */
	protected float globalBias;
	/**
	 * appender vector size: number of users + number of items + number of
	 * contextual conditions
	 */
	protected int numberOfFeatures;
	/**
	 * number of factors
	 */
	protected int numberOfFactors;

	/**
	 * weight vector
	 */
	protected DenseVector weightVector; // p
	/**
	 * parameter matrix(featureFactors)
	 */
	protected DenseMatrix featureFactors; // p x k
	/**
	 * parameter matrix(rateFactors)
	 */
	protected DenseMatrix actionFactors; // n x k
	/**
	 * regularization term for weight and factors
	 */
	protected float biasRegularization, weightRegularization, factorRegularization;

	/**
	 * init mean
	 */
	protected float initMean;

	/**
	 * init standard deviation
	 */
	protected float initStd;

	protected ContinuousProbability distribution;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// TODO 暂时不支持连续特征,考虑将连续特征离散化.
		int[] dimensions = new int[marker.getDiscreteOrder()];
		for (int order = 0; order < marker.getDiscreteOrder(); order++) {
			dimensions[order] = marker.getDiscreteAttribute(order).getSize();
		}
		this.trainTensor = SparseTensor.valueOf(dimensions, marker, (keyValue, instance) -> {
			int indexes[] = keyValue.getKey();
			for (int dimension = 0; dimension < indexes.length; dimension++) {
				indexes[dimension] = instance.getDiscreteFeature(dimension);
			}
			keyValue.setValue(instance.getMark());
		});

		// TODO 考虑重构,在AbstractRecommender初始化
		numberOfActions = trainTensor.getElementSize(); // set the number of
														// ratings
		maximumOfScore = configuration.getFloat("rec.recommender.maxrate", 12F);
		minimumOfScore = configuration.getFloat("rec.recommender.minrate", 0F);

		// initialize the parameters of FM
		for (int dimension = 0; dimension < trainTensor.getOrderSize(); dimension++) {
			numberOfFeatures += trainTensor.getDimensionSize(dimension); // set
																			// the
			// size
			// of
			// appender
			// vectors
		}

		numberOfFactors = configuration.getInteger("rec.factor.number");

		// init all weight with zero
		globalBias = 0;
		weightVector = DenseVector.valueOf(numberOfFeatures);

		// init factors with small value
		// TODO 此处需要重构
		initMean = configuration.getFloat("rec.init.mean", 0F);
		initStd = configuration.getFloat("rec.init.std", 0.1F);

		distribution = new ContinuousProbability(new NormalDistribution(new JDKRandomGenerator(0), initMean, initStd));
		featureFactors = DenseMatrix.valueOf(numberOfFeatures, numberOfFactors, MatrixMapper.distributionOf(distribution));

		biasRegularization = configuration.getFloat("rec.fm.regw0", 0.01f);
		weightRegularization = configuration.getFloat("rec.fm.regW", 0.01f);
		factorRegularization = configuration.getFloat("rec.fm.regF", 10f);
	}

	/**
	 * 获取特征向量
	 * 
	 * <pre>
	 * 实际为One Hot Encoding(一位有效编码)
	 * 详细原理与使用参考:http://blog.csdn.net/pipisorry/article/details/61193868
	 * </pre>
	 * 
	 * @param featureIndexes
	 * @return
	 */
	protected ArrayVector getFeatureVector(int[] featureIndexes) {
		int size = featureIndexes.length;
		int[] keys = new int[size];
		int cursor = 0;
		for (int index = 0; index < size; index++) {
			keys[index] += cursor + featureIndexes[index];
			cursor += trainTensor.getDimensionSize(index);
		}
		return new ArrayVector(numberOfFeatures, keys, (index, value, message) -> {
			return 1F;
		});
	}

	/**
	 * Predict the rating given a sparse appender vector.
	 * 
	 * @param userIndex
	 *            user Id
	 * @param itemIndex
	 *            item Id
	 * @param featureVector
	 *            the given vector to predict.
	 *
	 * @return predicted rating
	 * @throws RecommendationException
	 *             if error occurs
	 */
	protected float predict(DefaultScalar scalar, ArrayVector featureVector) {
		float value = 0;
		// global bias
		value += globalBias;
		// 1-way interaction
		value += scalar.dotProduct(weightVector, featureVector).getValue();

		// 2-way interaction
		for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
			float scoreSum = 0F;
			float predictSum = 0F;
			for (VectorScalar vectorTerm : featureVector) {
				float featureValue = vectorTerm.getValue();
				int featureIndex = vectorTerm.getIndex();
				float predictValue = featureFactors.getValue(featureIndex, factorIndex);

				scoreSum += predictValue * featureValue;
				predictSum += predictValue * predictValue * featureValue * featureValue;
			}
			value += (scoreSum * scoreSum - predictSum) / 2F;
		}

		return value;
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		// TODO 暂时不支持连续特征,考虑将连续特征离散化.
		ArrayVector featureVector = getFeatureVector(dicreteFeatures);
		return predict(scalar, featureVector);
	}

}
