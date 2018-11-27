package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.TreeSet;

import com.jstarcraft.core.utility.ReflectionUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.SymmetryMatrix;
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.exception.RecommendationException;
import com.jstarcraft.module.recommendation.recommender.ModelRecommender;
import com.jstarcraft.module.recommendation.utility.DriverUtility;
import com.jstarcraft.module.similarity.Similarity;

/**
 * Xia Ning and George Karypis, <strong>SLIM: Sparse Linear Methods for Top-N
 * Recommender Systems</strong>, ICDM 2011. <br>
 * <p>
 * Related Work:
 * <ul>
 * <li>Levy and Jack, Efficient Top-N Recommendation by Linear Regression, ISRS
 * 2013. This paper reports experimental results on the MovieLens (100K, 10M)
 * and Epinions datasets in terms of precision, MRR and HR@N (i.e.,
 * Recall@N).</li>
 * <li>Friedman et al., Regularization Paths for Generalized Linear Models via
 * Coordinate Descent, Journal of Statistical Software, 2010.</li>
 * </ul>
 *
 * @author guoguibing and Keqiang Wang
 */
public class SLIMRecommender extends ModelRecommender {

	/**
	 * W in original paper, a sparse matrix of aggregation coefficients
	 */
	// TODO 考虑修改为对称矩阵?
	private DenseMatrix coefficientMatrix;

	/**
	 * item's nearest neighbors for kNN > 0
	 */
	private int[][] itemNeighbors;

	/**
	 * regularization parameters for the L1 or L2 term
	 */
	private float regL1Norm, regL2Norm;

	/**
	 * number of nearest neighbors
	 */
	private int neighborSize;

	/**
	 * item similarity matrix
	 */
	private SymmetryMatrix similarityMatrix;

	private ArrayVector[] userVectors;

	private ArrayVector[] itemVectors;

	private Comparator<Entry<Integer, Double>> comparator = new Comparator<Entry<Integer, Double>>() {
		public int compare(Entry<Integer, Double> left, Entry<Integer, Double> right) {
			int value = -(left.getValue().compareTo(right.getValue()));
			if (value == 0) {
				value = left.getKey().compareTo(right.getKey());
			}
			return value;
		}
	};

	/**
	 * initialization
	 *
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		neighborSize = configuration.getInteger("rec.neighbors.knn.number", 50);
		regL1Norm = configuration.getFloat("rec.slim.regularization.l1", 1.0F);
		regL2Norm = configuration.getFloat("rec.slim.regularization.l2", 1.0F);

		// TODO 考虑重构
		coefficientMatrix = DenseMatrix.valueOf(numberOfItems, numberOfItems, MatrixMapper.RANDOM);
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			coefficientMatrix.setValue(itemIndex, itemIndex, 0F);
		}

		// initial guesses: make smaller guesses (e.g., W.init(0.01)) to speed
		// up training
		// TODO 修改为配置枚举
		Similarity similarity = ReflectionUtility.getInstance((Class<Similarity>) DriverUtility.getClass(configuration.getString("rec.similarity.class")));
		similarityMatrix = similarity.makeSimilarityMatrix(trainMatrix, true, configuration.getFloat("rec.similarity.shrinkage", 0F));

		// TODO 设置容量
		itemNeighbors = new int[numberOfItems][];
		HashMap<Integer, TreeSet<Entry<Integer, Double>>> itemNNs = new HashMap<>();
		for (MatrixScalar term : similarityMatrix) {
			int row = term.getRow();
			int column = term.getColumn();
			double value = term.getValue();
			// 忽略相似度为0的物品
			if (value == 0D) {
				continue;
			}
			TreeSet<Entry<Integer, Double>> neighbors = itemNNs.get(row);
			if (neighbors == null) {
				neighbors = new TreeSet<>(comparator);
				itemNNs.put(row, neighbors);
			}
			neighbors.add(new SimpleImmutableEntry<>(column, value));
			neighbors = itemNNs.get(column);
			if (neighbors == null) {
				neighbors = new TreeSet<>(comparator);
				itemNNs.put(column, neighbors);
			}
			neighbors.add(new SimpleImmutableEntry<>(row, value));
		}

		// 构建物品邻居映射
		for (Entry<Integer, TreeSet<Entry<Integer, Double>>> term : itemNNs.entrySet()) {
			TreeSet<Entry<Integer, Double>> neighbors = term.getValue();
			int[] value = new int[neighbors.size() < neighborSize ? neighbors.size() : neighborSize];
			int index = 0;
			for (Entry<Integer, Double> neighbor : neighbors) {
				value[index++] = neighbor.getKey();
				if (index >= neighborSize) {
					break;
				}
			}
			Arrays.sort(value);
			itemNeighbors[term.getKey()] = value;
		}

		userVectors = new ArrayVector[numberOfUsers];
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			userVectors[userIndex] = new ArrayVector(trainMatrix.getRowVector(userIndex), (index, value, message) -> {
				return value;
			});
		}

		itemVectors = new ArrayVector[numberOfItems];
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			itemVectors[itemIndex] = new ArrayVector(trainMatrix.getColumnVector(itemIndex), (index, value, message) -> {
				return value;
			});
		}
	}

	/**
	 * train model
	 *
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	protected void doPractice() {
		float[] rates = new float[numberOfUsers];
		// number of iteration cycles
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			// each cycle iterates through one coordinate direction
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				int[] neighborIndexes = itemNeighbors[itemIndex];
				if (neighborIndexes == null) {
					continue;
				}
				ArrayVector itemVector = itemVectors[itemIndex];
				for (VectorScalar term : itemVector) {
					rates[term.getIndex()] = term.getValue();
				}
				// for each nearest neighbor nearestNeighborItemIdx, update
				// coefficienMatrix by the coordinate
				// descent update rule
				for (int neighborIndex : neighborIndexes) {
					itemVector = itemVectors[neighborIndex];
					float valueSum = 0F, rateSum = 0F, errorSum = 0F;
					int count = itemVector.getElementSize();
					for (VectorScalar term : itemVector) {
						int userIndex = term.getIndex();
						float neighborRate = term.getValue();
						float userRate = rates[userIndex];
						float error = userRate - predict(userIndex, itemIndex, neighborIndexes, neighborIndex);
						valueSum += neighborRate * error;
						rateSum += neighborRate * neighborRate;
						errorSum += error * error;
					}
					valueSum /= count;
					rateSum /= count;
					errorSum /= count;
					// TODO 此处考虑重构
					float coefficient = coefficientMatrix.getValue(neighborIndex, itemIndex);
					totalLoss += errorSum + 0.5F * regL2Norm * coefficient * coefficient + regL1Norm * coefficient;
					if (regL1Norm < Math.abs(valueSum)) {
						if (valueSum > 0) {
							coefficient = (valueSum - regL1Norm) / (regL2Norm + rateSum);
						} else {
							// One doubt: in this case, wij<0, however, the
							// paper says wij>=0. How to gaurantee that?
							coefficient = (valueSum + regL1Norm) / (regL2Norm + rateSum);
						}
					} else {
						coefficient = 0F;
					}
					coefficientMatrix.setValue(neighborIndex, itemIndex, coefficient);
				}
				itemVector = itemVectors[itemIndex];
				for (VectorScalar term : itemVector) {
					rates[term.getIndex()] = 0F;
				}
			}

			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			currentLoss = totalLoss;
		}
	}

	/**
	 * predict a specific ranking score for user userIdx on item itemIdx.
	 *
	 * @param userIndex
	 *            user index
	 * @param itemIndex
	 *            item index
	 * @param excludIndex
	 *            excluded item index
	 * @return a prediction without the contribution of excluded item
	 */
	private float predict(int userIndex, int itemIndex, int[] neighbors, int currentIndex) {
		float value = 0F;
		ArrayVector userVector = userVectors[userIndex];
		if (userVector.getElementSize() == 0) {
			return value;
		}
		int leftIndex = 0, rightIndex = 0, leftSize = userVector.getElementSize(), rightSize = neighbors.length;
		Iterator<VectorScalar> iterator = userVector.iterator();
		VectorScalar term = iterator.next();
		// 判断两个有序数组中是否存在相同的数字
		while (leftIndex < leftSize && rightIndex < rightSize) {
			if (term.getIndex() == neighbors[rightIndex]) {
				if (neighbors[rightIndex] != currentIndex) {
					value += term.getValue() * coefficientMatrix.getValue(neighbors[rightIndex], itemIndex);
				}
				if (iterator.hasNext()) {
					term = iterator.next();
				}
				leftIndex++;
				rightIndex++;
			} else if (term.getIndex() > neighbors[rightIndex]) {
				rightIndex++;
			} else if (term.getIndex() < neighbors[rightIndex]) {
				if (iterator.hasNext()) {
					term = iterator.next();
				}
				leftIndex++;
			}
		}
		return value;
	}

	/**
	 * predict a specific ranking score for user userIdx on item itemIdx.
	 *
	 * @param userIndex
	 *            user index
	 * @param itemIndex
	 *            item index
	 * @return predictive ranking score for user userIdx on item itemIdx
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		int[] neighbors = itemNeighbors[itemIndex];
		if (neighbors == null) {
			return 0F;
		}
		return predict(userIndex, itemIndex, neighbors, -1);
	}

}
