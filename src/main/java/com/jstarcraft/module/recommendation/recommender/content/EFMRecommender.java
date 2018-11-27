package com.jstarcraft.module.recommendation.recommender.content;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.core.utility.StringUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.DataSample;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * EFM Recommender Zhang Y, Lai G, Zhang M, et al. Explicit factor models for
 * explainable recommendation based on phrase-level sentiment analysis[C]
 * {@code Proceedings of the 37th international ACM SIGIR conference on Research & development in information retrieval.  ACM, 2014: 83-92}.
 *
 * @author ChenXu and SunYatong
 */
public abstract class EFMRecommender extends MatrixFactorizationRecommender {

	protected String commentField;
	protected int commentDimension;
	protected int numberOfFeatures;
	protected int numberOfExplicitFeatures;
	protected int numberOfImplicitFeatures;
	protected float scoreScale;
	protected DenseMatrix featureFactors;
	protected DenseMatrix userExplicitFactors;
	protected DenseMatrix userImplicitFactors;
	protected DenseMatrix itemExplicitFactors;
	protected DenseMatrix itemImplicitFactors;
	protected SparseMatrix userFeatures;
	protected SparseMatrix itemFeatures;
	protected float attentionRegularization;
	protected float qualityRegularization;
	protected float explicitRegularization;
	protected float implicitRegularization;
	protected float featureRegularization;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);

		commentField = configuration.getString("data.model.fields.comment");
		commentDimension = model.getDiscreteDimension(commentField);
		Object[] wordValues = model.getDiscreteAttribute(commentDimension).getDatas();

		scoreScale = maximumOfScore - minimumOfScore;
		numberOfExplicitFeatures = configuration.getInteger("rec.factor.explicit", 5);
		numberOfImplicitFeatures = numberOfFactors - numberOfExplicitFeatures;
		attentionRegularization = configuration.getFloat("rec.regularization.lambdax", 0.001F);
		qualityRegularization = configuration.getFloat("rec.regularization.lambday", 0.001F);
		explicitRegularization = configuration.getFloat("rec.regularization.lambdau", 0.001F);
		implicitRegularization = configuration.getFloat("rec.regularization.lambdah", 0.001F);
		featureRegularization = configuration.getFloat("rec.regularization.lambdav", 0.001F);

		Map<String, Integer> featureDictionaries = new HashMap<>();
		Map<Integer, StringBuilder> userDictionaries = new HashMap<>();
		Map<Integer, StringBuilder> itemDictionaries = new HashMap<>();

		numberOfFeatures = 0;
		// // TODO 此处保证所有特征都会被识别
		// for (Object value : wordValues) {
		// String wordValue = (String) value;
		// String[] words = wordValue.split(" ");
		// for (String word : words) {
		// // TODO 此处似乎是Bug,不应该再将word划分为更细粒度.
		// String feature = word.split(":")[0];
		// if (!featureDictionaries.containsKey(feature) &&
		// StringUtils.isNotEmpty(feature)) {
		// featureDictionaries.put(feature, numberOfWords);
		// numberOfWords++;
		// }
		// }
		// }

		for (DataSample sample : marker) {
			int userIndex = sample.getDiscreteFeature(userDimension);
			int itemIndex = sample.getDiscreteFeature(itemDimension);
			int wordIndex = sample.getDiscreteFeature(commentDimension);
			String wordValue = (String) wordValues[wordIndex];
			String[] words = wordValue.split(" ");
			StringBuilder buffer;
			for (String word : words) {
				// TODO 此处似乎是Bug,不应该再将word划分为更细粒度.
				String feature = word.split(":")[0];
				if (!featureDictionaries.containsKey(feature) && !StringUtility.isEmpty(feature)) {
					featureDictionaries.put(feature, numberOfFeatures++);
				}
				buffer = userDictionaries.get(userIndex);
				if (buffer != null) {
					buffer.append(" ").append(word);
				} else {
					userDictionaries.put(userIndex, new StringBuilder(word));
				}
				buffer = itemDictionaries.get(itemIndex);
				if (buffer != null) {
					buffer.append(" ").append(word);
				} else {
					itemDictionaries.put(itemIndex, new StringBuilder(word));
				}
			}
		}

		// Create V,U1,H1,U2,H2
		featureFactors = DenseMatrix.valueOf(numberOfFeatures, numberOfExplicitFeatures, MatrixMapper.randomOf(0.01F));
		userExplicitFactors = DenseMatrix.valueOf(numberOfUsers, numberOfExplicitFeatures, MatrixMapper.RANDOM);
		userImplicitFactors = DenseMatrix.valueOf(numberOfUsers, numberOfImplicitFeatures, MatrixMapper.RANDOM);
		itemExplicitFactors = DenseMatrix.valueOf(numberOfItems, numberOfExplicitFeatures, MatrixMapper.RANDOM);
		itemImplicitFactors = DenseMatrix.valueOf(numberOfItems, numberOfImplicitFeatures, MatrixMapper.RANDOM);

		float[] featureValues = new float[numberOfFeatures];

		// compute UserFeatureAttention
		Table<Integer, Integer, Float> userTable = HashBasedTable.create();
		for (Entry<Integer, StringBuilder> term : userDictionaries.entrySet()) {
			int userIndex = term.getKey();
			String[] words = term.getValue().toString().split(" ");
			for (String word : words) {
				if (!StringUtility.isEmpty(word)) {
					int featureIndex = featureDictionaries.get(word.split(":")[0]);
					featureValues[featureIndex] += 1F;
				}
			}
			for (int featureIndex = 0; featureIndex < numberOfFeatures; featureIndex++) {
				if (featureValues[featureIndex] != 0F) {
					float value = (float) (1F + (scoreScale - 1F) * (2F / (1F + Math.exp(-featureValues[featureIndex])) - 1F));
					userTable.put(userIndex, featureIndex, value);
					featureValues[featureIndex] = 0F;
				}
			}
		}
		userFeatures = SparseMatrix.valueOf(numberOfUsers, numberOfFeatures, userTable);
		// compute ItemFeatureQuality
		Table<Integer, Integer, Float> itemTable = HashBasedTable.create();
		for (Entry<Integer, StringBuilder> term : itemDictionaries.entrySet()) {
			int itemIndex = term.getKey();
			String[] words = term.getValue().toString().split(" ");
			for (String word : words) {
				if (!StringUtility.isEmpty(word)) {
					int featureIndex = featureDictionaries.get(word.split(":")[0]);
					featureValues[featureIndex] += Double.parseDouble(word.split(":")[1]);
				}
			}
			for (int featureIndex = 0; featureIndex < numberOfFeatures; featureIndex++) {
				if (featureValues[featureIndex] != 0F) {
					float value = (float) (1F + (scoreScale - 1F) / (1F + Math.exp(-featureValues[featureIndex])));
					itemTable.put(itemIndex, featureIndex, value);
					featureValues[featureIndex] = 0F;
				}
			}
		}
		itemFeatures = SparseMatrix.valueOf(numberOfItems, numberOfFeatures, itemTable);

		logger.info("numUsers:" + numberOfUsers);
		logger.info("numItems:" + numberOfItems);
		logger.info("numFeatures:" + numberOfFeatures);
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			for (int featureIndex = 0; featureIndex < numberOfFeatures; featureIndex++) {
				if (userFeatures.getColumnScope(featureIndex) > 0 && itemFeatures.getColumnScope(featureIndex) > 0) {
					SparseVector userVector = userFeatures.getColumnVector(featureIndex);
					SparseVector itemVector = itemFeatures.getColumnVector(featureIndex);
					// TODO 此处需要重构,应该避免不断构建SparseVector.
					int feature = featureIndex;
					ArrayVector userFactors = new ArrayVector(userVector, (index, value, message) -> {
						return predictUserFactor(scalar, index, feature);
					});
					ArrayVector itemFactors = new ArrayVector(itemVector, (index, value, message) -> {
						return predictItemFactor(scalar, index, feature);
					});
					for (int factorIndex = 0; factorIndex < numberOfExplicitFeatures; factorIndex++) {
						DenseVector factorUsersVector = userExplicitFactors.getColumnVector(factorIndex);
						DenseVector factorItemsVector = itemExplicitFactors.getColumnVector(factorIndex);
						float numerator = attentionRegularization * scalar.dotProduct(factorUsersVector, userVector).getValue() + qualityRegularization * scalar.dotProduct(factorItemsVector, itemVector).getValue();
						float denominator = attentionRegularization * scalar.dotProduct(factorUsersVector, userFactors).getValue() + qualityRegularization * scalar.dotProduct(factorItemsVector, itemFactors).getValue() + featureRegularization * featureFactors.getValue(featureIndex, factorIndex) + MathUtility.EPSILON;
						featureFactors.setValue(featureIndex, factorIndex, (float) (featureFactors.getValue(featureIndex, factorIndex) * Math.sqrt(numerator / denominator)));
					}
				}
			}

			// Update UserFeatureMatrix by fixing the others
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				if (trainMatrix.getRowScope(userIndex) > 0 && userFeatures.getRowScope(userIndex) > 0) {
					SparseVector userVector = trainMatrix.getRowVector(userIndex);
					SparseVector attentionVector = userFeatures.getRowVector(userIndex);
					// TODO 此处需要重构,应该避免不断构建SparseVector.
					int user = userIndex;
					ArrayVector itemPredictsVector = new ArrayVector(userVector, (index, value, message) -> {
						return predict(user, index);
					});
					ArrayVector attentionPredVec = new ArrayVector(attentionVector, (index, value, message) -> {
						return predictUserFactor(scalar, user, index);
					});
					for (int factorIndex = 0; factorIndex < numberOfExplicitFeatures; factorIndex++) {
						DenseVector factorItemsVector = itemExplicitFactors.getColumnVector(factorIndex);
						DenseVector featureVector = featureFactors.getColumnVector(factorIndex);
						float numerator = scalar.dotProduct(factorItemsVector, userVector).getValue() + attentionRegularization * scalar.dotProduct(featureVector, attentionVector).getValue();
						float denominator = scalar.dotProduct(factorItemsVector, itemPredictsVector).getValue() + attentionRegularization * scalar.dotProduct(featureVector, attentionPredVec).getValue() + explicitRegularization * userExplicitFactors.getValue(userIndex, factorIndex) + MathUtility.EPSILON;
						userExplicitFactors.setValue(userIndex, factorIndex, (float) (userExplicitFactors.getValue(userIndex, factorIndex) * Math.sqrt(numerator / denominator)));
					}
				}
			}

			// Update ItemFeatureMatrix by fixing the others
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				if (trainMatrix.getColumnScope(itemIndex) > 0 && itemFeatures.getRowScope(itemIndex) > 0) {
					SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
					SparseVector qualityVector = itemFeatures.getRowVector(itemIndex);
					// TODO 此处需要重构,应该避免不断构建SparseVector.
					int item = itemIndex;
					ArrayVector userPredictsVector = new ArrayVector(itemVector, (index, value, message) -> {
						return predict(index, item);
					});
					ArrayVector qualityPredVec = new ArrayVector(qualityVector, (index, value, message) -> {
						return predictItemFactor(scalar, item, index);
					});
					for (int factorIndex = 0; factorIndex < numberOfExplicitFeatures; factorIndex++) {
						DenseVector factorUsersVector = userExplicitFactors.getColumnVector(factorIndex);
						DenseVector featureVector = featureFactors.getColumnVector(factorIndex);
						float numerator = scalar.dotProduct(factorUsersVector, itemVector).getValue() + qualityRegularization * scalar.dotProduct(featureVector, qualityVector).getValue();
						float denominator = scalar.dotProduct(factorUsersVector, userPredictsVector).getValue() + qualityRegularization * scalar.dotProduct(featureVector, qualityPredVec).getValue() + explicitRegularization * itemExplicitFactors.getValue(itemIndex, factorIndex) + MathUtility.EPSILON;
						itemExplicitFactors.setValue(itemIndex, factorIndex, (float) (itemExplicitFactors.getValue(itemIndex, factorIndex) * Math.sqrt(numerator / denominator)));
					}
				}
			}

			// Update UserHiddenMatrix by fixing the others
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				if (trainMatrix.getRowScope(userIndex) > 0) {
					SparseVector userVector = trainMatrix.getRowVector(userIndex);
					// TODO 此处需要重构,应该避免不断构建SparseVector.
					int user = userIndex;
					ArrayVector itemPredictsVector = new ArrayVector(userVector, (index, value, message) -> {
						return predict(user, index);
					});
					for (int factorIndex = 0; factorIndex < numberOfImplicitFeatures; factorIndex++) {
						DenseVector hiddenItemsVector = itemImplicitFactors.getColumnVector(factorIndex);
						float numerator = scalar.dotProduct(hiddenItemsVector, userVector).getValue();
						float denominator = scalar.dotProduct(hiddenItemsVector, itemPredictsVector).getValue() + implicitRegularization * userImplicitFactors.getValue(userIndex, factorIndex) + MathUtility.EPSILON;
						userImplicitFactors.setValue(userIndex, factorIndex, (float) (userImplicitFactors.getValue(userIndex, factorIndex) * Math.sqrt(numerator / denominator)));
					}
				}
			}

			// Update ItemHiddenMatrix by fixing the others
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				if (trainMatrix.getColumnScope(itemIndex) > 0) {
					SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
					// TODO 此处需要重构,应该避免不断构建SparseVector.
					int item = itemIndex;
					ArrayVector userPredictsVector = new ArrayVector(itemVector, (index, value, message) -> {
						return predict(index, item);
					});
					for (int factorIndex = 0; factorIndex < numberOfImplicitFeatures; factorIndex++) {
						DenseVector hiddenUsersVector = userImplicitFactors.getColumnVector(factorIndex);
						float numerator = scalar.dotProduct(hiddenUsersVector, itemVector).getValue();
						float denominator = scalar.dotProduct(hiddenUsersVector, userPredictsVector).getValue() + implicitRegularization * itemImplicitFactors.getValue(itemIndex, factorIndex) + MathUtility.EPSILON;
						itemImplicitFactors.setValue(itemIndex, factorIndex, (float) (itemImplicitFactors.getValue(itemIndex, factorIndex) * Math.sqrt(numerator / denominator)));
					}
				}
			}

			// Compute loss value
			for (MatrixScalar term : trainMatrix) {
				int userIndex = term.getRow();
				int itemIndex = term.getColumn();
				double rating = term.getValue();
				double predRating = scalar.dotProduct(userExplicitFactors.getRowVector(userIndex), itemExplicitFactors.getRowVector(itemIndex)).getValue() + scalar.dotProduct(userImplicitFactors.getRowVector(userIndex), itemImplicitFactors.getRowVector(itemIndex)).getValue();
				totalLoss += (rating - predRating) * (rating - predRating);
			}

			for (MatrixScalar term : userFeatures) {
				int userIndex = term.getRow();
				int featureIndex = term.getColumn();
				double real = term.getValue();
				double pred = predictUserFactor(scalar, userIndex, featureIndex);
				totalLoss += (real - pred) * (real - pred);
			}

			for (MatrixScalar term : itemFeatures) {
				int itemIndex = term.getRow();
				int featureIndex = term.getColumn();
				double real = term.getValue();
				double pred = predictItemFactor(scalar, itemIndex, featureIndex);
				totalLoss += (real - pred) * (real - pred);
			}

			totalLoss += explicitRegularization * (Math.pow(userExplicitFactors.getNorm(2), 2) + Math.pow(itemExplicitFactors.getNorm(2), 2));
			totalLoss += implicitRegularization * (Math.pow(userImplicitFactors.getNorm(2), 2) + Math.pow(itemImplicitFactors.getNorm(2), 2));
			totalLoss += featureRegularization * Math.pow(featureFactors.getNorm(2), 2);

			logger.info("iter:" + iterationStep + ", loss:" + totalLoss);
		}
	}

	protected float predictUserFactor(DefaultScalar scalar, int userIndex, int featureIndex) {
		return scalar.dotProduct(userExplicitFactors.getRowVector(userIndex), featureFactors.getRowVector(featureIndex)).getValue();
	}

	protected float predictItemFactor(DefaultScalar scalar, int itemIndex, int featureIndex) {
		return scalar.dotProduct(itemExplicitFactors.getRowVector(itemIndex), featureFactors.getRowVector(featureIndex)).getValue();
	}

	@Override
	protected float predict(int userIndex, int itemIndex) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		return scalar.dotProduct(userExplicitFactors.getRowVector(userIndex), itemExplicitFactors.getRowVector(itemIndex)).getValue() + scalar.dotProduct(userImplicitFactors.getRowVector(userIndex), itemImplicitFactors.getRowVector(itemIndex)).getValue();
	}

}
