package com.jstarcraft.module.data.splitter;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSplitter;
import com.jstarcraft.module.data.IntegerArray;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.processor.DataMatcher;

/**
 * 随机处理器
 * 
 * @author Birdy
 *
 */
public class RandomSplitter implements DataSplitter {

	private InstanceAccessor dataModel;

	private IntegerArray trainReference;

	private IntegerArray testReference;

	public RandomSplitter(InstanceAccessor model, String matchField, double random) {
		dataModel = model;
		int size = model.getSize();
		int[] paginations;
		int[] positions = new int[size];
		for (int index = 0; index < size; index++) {
			positions[index] = index;
		}
		int matchDimension = model.getDiscreteDimension(matchField);
		paginations = new int[model.getDiscreteAttribute(matchDimension).getSize() + 1];
		DataMatcher matcher = DataMatcher.discreteOf(model, matchDimension);
		matcher.match(paginations, positions);

		trainReference = new IntegerArray();
		testReference = new IntegerArray();
		size = paginations.length - 1;
		for (int index = 0; index < size; index++) {
			for (int from = paginations[index], to = paginations[index + 1]; from < to; from++) {
				if (RandomUtility.randomDouble(1D) < random) {
					trainReference.associateData(positions[from]);
				} else {
					testReference.associateData(positions[from]);
				}
			}
		}
	}

	@Override
	public int getSize() {
		return 1;
	}

	@Override
	public InstanceAccessor getDataModel() {
		return dataModel;
	}

	@Override
	public IntegerArray getTrainReference(int index) {
		return trainReference;
	}

	@Override
	public IntegerArray getTestReference(int index) {
		return testReference;
	}

}