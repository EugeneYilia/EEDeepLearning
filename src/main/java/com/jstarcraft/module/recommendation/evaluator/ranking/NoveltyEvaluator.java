/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
package com.jstarcraft.module.recommendation.evaluator.ranking;

import java.util.Collection;
import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.evaluator.RankingEvaluator;

/**
 * NoveltyEvaluator
 * 
 * Often also called 'Mean Self-Information' or Surprisal
 * 
 * Look at Section '4.2.5 Novelty' of article:
 * 
 * Javari, Amin, and Mahdi Jalili. "A probabilistic model to resolve
 * diversity–accuracy challenge of recommendation systems." Knowledge and
 * Information Systems 44.3 (2015): 609-627.
 * 
 * 
 * Calculates Self-Information of each recommender result list. And then
 * calculates the average of this of all result lists in test set.
 * 
 * But please take also attention to the assumed probability space:
 * 
 * The probability of an item is assumed to be the purchase probability.
 * (Estimated by items purchased divided by all items purchased.) Surely there
 * is also independence assumed between items.
 * 
 * This assumption about the probability space is different from the
 * EntropyEvaluator
 * 
 *
 * @author Daniel Velten, Karlsruhe, Germany, SunYatong
 */
public class NoveltyEvaluator extends RankingEvaluator {

	private int numberOfUsers;

	private int[] itemCounts;

	public NoveltyEvaluator(int size, SparseMatrix dataMatrix) {
		super(size);
		// use the purchase counts of the train and test data set
		numberOfUsers = dataMatrix.getRowSize();
		int numberOfItems = dataMatrix.getColumnSize();
		itemCounts = new int[numberOfItems];
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			itemCounts[itemIndex] = dataMatrix.getColumnScope(itemIndex);
		}
	}

	/**
	 * Evaluate on the test set with the the list of recommended items.
	 *
	 * @param testMatrix
	 *            the given test set
	 * @param recommendedList
	 *            the list of recommended items
	 * @return evaluate result
	 */
	@Override
	protected float measure(Collection<Integer> checkCollection, List<KeyValue<Integer, Float>> recommendList) {
		if (recommendList.size() > size) {
			recommendList = recommendList.subList(0, size);
		}

		float sum = 0F;
		for (KeyValue<Integer, Float> keyValue : recommendList) {
			int itemIndex = keyValue.getKey();
			int count = itemCounts[itemIndex];
			if (count > 0) {
				float probability = ((float) count) / numberOfUsers;
				float entropy = (float) -Math.log(probability);
				sum += entropy;
			}
		}
		return (float) (sum / Math.log(2F));
	}

}
