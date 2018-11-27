package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.DataInstance;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

public class RankVFCDRecommender extends MatrixFactorizationRecommender {

	/**
	 * two low-rank item matrices, an item-item similarity was learned as a product
	 * of these two matrices
	 */
	private DenseMatrix userFactors, explicitItemFactors;
	private float alpha, beta, gamma, lamutaE;
	private SparseMatrix featureMatrix;
	private DenseVector featureVector;
	private int numberOfFeatures;
	private DenseMatrix featureFactors, implicitItemFactors, factorMatrix;
	private SparseMatrix relationMatrix;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);

		// TODO 此处代码可以消除(使用常量Marker代替或者使用binarize.threshold)
		for (MatrixScalar term : trainMatrix) {
			term.setValue(1F);
		}

		alpha = configuration.getFloat("rec.rankvfcd.alpha", 5F);
		beta = configuration.getFloat("rec.rankvfcd.beta", 10F);
		gamma = configuration.getFloat("rec.rankvfcd.gamma", 50F);
		lamutaE = configuration.getFloat("rec.rankvfcd.lamutaE", 50F);
		numberOfFeatures = 4096;

		userFactors = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.distributionOf(distribution));
		explicitItemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.distributionOf(distribution));
		implicitItemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.distributionOf(distribution));
		featureFactors = DenseMatrix.valueOf(numberOfFeatures, numberOfFactors, MatrixMapper.distributionOf(distribution));

		// 相关矩阵
		InstanceAccessor relationModel = space.getModule("relation");
		Table<Integer, Integer, Float> relationTable = HashBasedTable.create();
		for (DataInstance instance : relationModel) {
			int itemIndex = instance.getDiscreteFeature(0);
			int neighborIndex = instance.getDiscreteFeature(1);
			relationTable.put(itemIndex, neighborIndex, 1F);
		}
		relationMatrix = SparseMatrix.valueOf(numberOfItems, numberOfItems, relationTable);
		relationTable = null;

		// 特征矩阵
		float minimumValue = Float.MAX_VALUE;
		float maximumValue = Float.MIN_VALUE;
		Table<Integer, Integer, Float> visualTable = HashBasedTable.create();
		InstanceAccessor featureModel = space.getModule("article");
		for (DataInstance instance : featureModel) {
			int itemIndex = instance.getDiscreteFeature(0);
			int featureIndex = instance.getDiscreteFeature(1);
			Float featureValue = instance.getContinuousFeature(0);
			if (featureValue < minimumValue) {
				minimumValue = featureValue;
			}
			if (featureValue > maximumValue) {
				maximumValue = featureValue;
			}
			visualTable.put(featureIndex, itemIndex, featureValue);
		}
		featureMatrix = SparseMatrix.valueOf(numberOfFeatures, numberOfItems, visualTable);
		visualTable = null;
		featureMatrix.normalize(minimumValue, maximumValue);

		factorMatrix = DenseMatrix.valueOf(numberOfFactors, numberOfItems);
		featureVector = DenseVector.valueOf(numberOfFeatures);
		for (MatrixScalar term : featureMatrix) {
			int featureIndex = term.getRow();
			float value = featureVector.getValue(featureIndex) + term.getValue() * term.getValue();
			featureVector.setValue(featureIndex, value);
		}
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		// Init caches
		float[] prediction_users = new float[numberOfUsers];
		float[] prediction_items = new float[numberOfItems];
		float[] prediction_itemrelated = new float[numberOfItems];
		float[] prediction_relateditem = new float[numberOfItems];
		float[] w_users = new float[numberOfUsers];
		float[] w_items = new float[numberOfItems];
		float[] q_itemrelated = new float[numberOfItems];
		float[] q_relateditem = new float[numberOfItems];

		DenseMatrix explicitItemDeltas = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
		DenseMatrix implicitItemDeltas = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
		DenseMatrix userDeltas = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);

		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			// Update the Sq cache
			explicitItemDeltas.dotProduct(explicitItemFactors, true, explicitItemFactors, false, MathCalculator.SERIAL);
			// Step 1: update user factors;
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				SparseVector userVector = trainMatrix.getRowVector(userIndex);
				for (VectorScalar term : userVector) {
					int itemIndex = term.getIndex();
					prediction_items[itemIndex] = scalar.dotProduct(userFactors.getRowVector(userIndex), explicitItemFactors.getRowVector(itemIndex)).getValue();
					w_items[itemIndex] = 1F + alpha * term.getValue();
				}
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float numerator = 0F, denominator = userRegularization + explicitItemDeltas.getValue(factorIndex, factorIndex);
					// TODO 此处可以改为减法
					for (int k = 0; k < numberOfFactors; k++) {
						if (factorIndex != k) {
							numerator -= userFactors.getValue(userIndex, k) * explicitItemDeltas.getValue(factorIndex, k);
						}
					}
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					for (VectorScalar entry : userVector) {
						int i = entry.getIndex();
						float qif = explicitItemFactors.getValue(i, factorIndex);
						prediction_items[i] -= userFactor * qif;
						numerator += (w_items[i] - (w_items[i] - 1) * prediction_items[i]) * qif;
						denominator += (w_items[i] - 1) * qif * qif;
					}
					// update puf
					userFactor = numerator / denominator;
					userFactors.setValue(userIndex, factorIndex, userFactor);
					for (VectorScalar term : userVector) {
						int itemIndex = term.getIndex();
						prediction_items[itemIndex] += userFactor * explicitItemFactors.getValue(itemIndex, factorIndex);
					}
				}
			}

			// Update the Sp cache
			userDeltas.dotProduct(userFactors, true, userFactors, false, MathCalculator.SERIAL);
			implicitItemDeltas.dotProduct(implicitItemFactors, true, implicitItemFactors, false, MathCalculator.SERIAL);
			DenseMatrix ETF = factorMatrix;
			ETF.dotProduct(featureFactors, true, featureMatrix, false, MathCalculator.PARALLEL);
			// Step 2: update item factors;
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
				SparseVector relationVector = relationMatrix.getRowVector(itemIndex);
				for (VectorScalar term : itemVector) {
					int userIndex = term.getIndex();

					prediction_users[userIndex] = scalar.dotProduct(userFactors.getRowVector(userIndex), explicitItemFactors.getRowVector(itemIndex)).getValue();
					w_users[userIndex] = 1F + alpha * term.getValue();
				}
				for (VectorScalar term : relationVector) {
					int neighborIndex = term.getIndex();
					prediction_itemrelated[neighborIndex] = scalar.dotProduct(explicitItemFactors.getRowVector(itemIndex), implicitItemFactors.getRowVector(neighborIndex)).getValue();
					q_itemrelated[neighborIndex] = 1F + alpha * term.getValue();
				}
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float explicitNumerator = 0F, explicitDenominator = userDeltas.getValue(factorIndex, factorIndex) + itemRegularization;
					float implicitNumerator = 0F, implicitDenominator = implicitItemDeltas.getValue(factorIndex, factorIndex);
					// TODO 此处可以改为减法
					for (int k = 0; k < numberOfFactors; k++) {
						if (factorIndex != k) {
							explicitNumerator -= explicitItemFactors.getValue(itemIndex, k) * userDeltas.getValue(k, factorIndex);
							implicitNumerator -= explicitItemFactors.getValue(itemIndex, k) * implicitItemDeltas.getValue(k, factorIndex);
						}
					}
					float explicitItemFactor = explicitItemFactors.getValue(itemIndex, factorIndex);
					for (VectorScalar term : itemVector) {
						int userIndex = term.getIndex();
						float userFactor = userFactors.getValue(userIndex, factorIndex);
						prediction_users[userIndex] -= userFactor * explicitItemFactor;
						explicitNumerator += (w_users[userIndex] - (w_users[userIndex] - 1) * prediction_users[userIndex]) * userFactor;
						explicitDenominator += (w_users[userIndex] - 1) * userFactor * userFactor;
					}
					for (VectorScalar term : relationVector) {
						int neighborIndex = term.getIndex();
						float implicitItemFactor = implicitItemFactors.getValue(neighborIndex, factorIndex);
						prediction_itemrelated[neighborIndex] -= implicitItemFactor * explicitItemFactor;
						implicitNumerator += (q_itemrelated[neighborIndex] - (q_itemrelated[neighborIndex] - 1) * prediction_itemrelated[neighborIndex]) * implicitItemFactor;
						implicitDenominator += (q_itemrelated[neighborIndex] - 1) * implicitItemFactor * implicitItemFactor;
					}
					// update qif
					explicitItemFactor = (explicitNumerator + implicitNumerator * beta + gamma * ETF.getValue(factorIndex, itemIndex)) / (explicitDenominator + implicitDenominator * beta + gamma);
					explicitItemFactors.setValue(itemIndex, factorIndex, explicitItemFactor);
					for (VectorScalar term : itemVector) {
						int userIndex = term.getIndex();
						prediction_users[userIndex] += userFactors.getValue(userIndex, factorIndex) * explicitItemFactor;
					}
					for (VectorScalar term : relationVector) {
						int neighborIndex = term.getIndex();
						prediction_itemrelated[neighborIndex] += implicitItemFactors.getValue(neighborIndex, factorIndex) * explicitItemFactor;
					}
				}
			}

			explicitItemDeltas.dotProduct(explicitItemFactors, true, explicitItemFactors, false, MathCalculator.SERIAL);
			// Step 1: update Z factors;
			for (int neighborIndex = 0; neighborIndex < numberOfItems; neighborIndex++) {
				SparseVector relationVector = relationMatrix.getColumnVector(neighborIndex);
				for (VectorScalar term : relationVector) {
					int itemIndex = term.getIndex();
					prediction_relateditem[itemIndex] = scalar.dotProduct(explicitItemFactors.getRowVector(itemIndex), implicitItemFactors.getRowVector(neighborIndex)).getValue();
					q_relateditem[itemIndex] = 1F + alpha * term.getValue();
				}
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float numerator = 0F, denominator = explicitItemDeltas.getValue(factorIndex, factorIndex);
					// TODO 此处可以改为减法
					for (int k = 0; k < numberOfFactors; k++) {
						if (factorIndex != k) {
							numerator -= implicitItemFactors.getValue(neighborIndex, k) * explicitItemDeltas.getValue(factorIndex, k);
						}
					}
					float implicitItemFactor = implicitItemFactors.getValue(neighborIndex, factorIndex);
					for (VectorScalar term : relationVector) {
						int itemIndex = term.getIndex();
						float explicitItemFactor = explicitItemFactors.getValue(itemIndex, factorIndex);
						prediction_relateditem[itemIndex] -= implicitItemFactor * explicitItemFactor;
						numerator += (q_relateditem[itemIndex] - (q_relateditem[itemIndex] - 1) * prediction_relateditem[itemIndex]) * explicitItemFactor;
						denominator += (q_relateditem[itemIndex] - 1) * explicitItemFactor * explicitItemFactor;
					}
					// update puf
					implicitItemFactor = beta * numerator / (beta * denominator + itemRegularization);
					implicitItemFactors.setValue(neighborIndex, factorIndex, implicitItemFactor);
					for (VectorScalar term : relationVector) {
						int itemIndex = term.getIndex();
						prediction_relateditem[itemIndex] += implicitItemFactor * explicitItemFactors.getValue(itemIndex, factorIndex);
					}
				}
			}

			for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
				for (int featureIndex = 0; featureIndex < numberOfFeatures; featureIndex++) {
					SparseVector featureVector = featureMatrix.getRowVector(featureIndex);
					float numerator = 0F, denominator = featureFactors.getValue(featureIndex, factorIndex);
					for (VectorScalar term : featureVector) {
						float featureValue = term.getValue();
						int itemIndex = term.getIndex();
						ETF.setValue(factorIndex, itemIndex, ETF.getValue(factorIndex, itemIndex) - denominator * featureValue);
						numerator += (explicitItemFactors.getValue(itemIndex, factorIndex) - ETF.getValue(factorIndex, itemIndex)) * featureValue;
					}
					denominator = numerator * gamma / (gamma * this.featureVector.getValue(featureIndex) + lamutaE);
					featureFactors.setValue(featureIndex, factorIndex, denominator);
					for (VectorScalar term : featureVector) {
						float featureValue = term.getValue();
						int itemIndex = term.getIndex();
						ETF.setValue(factorIndex, itemIndex, ETF.getValue(factorIndex, itemIndex) + denominator * featureValue);
					}
				}
			}
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			currentLoss = totalLoss;
			// TODO 目前没有totalLoss.
		}
		factorMatrix.dotProduct(featureFactors, true, featureMatrix, false, MathCalculator.PARALLEL);
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float score = 0F;
		if (trainMatrix.getColumnVector(itemIndex).getElementSize() == 0) {
			score = scalar.dotProduct(userFactors.getRowVector(userIndex), factorMatrix.getColumnVector(itemIndex)).getValue();
		} else {
			score = scalar.dotProduct(userFactors.getRowVector(userIndex), explicitItemFactors.getRowVector(itemIndex)).getValue();
		}
		return score;
	}

}
