package com.jstarcraft.module.data.splitter;

import com.jstarcraft.module.data.DataSplitter;
import com.jstarcraft.module.data.IntegerArray;
import com.jstarcraft.module.data.accessor.DataInstance;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.processor.DataSelector;

/**
 * 指定实例处理器
 * 
 * @author Birdy
 *
 */
// TODO 准备改名为SpecificInstanceSplitter
public class GivenInstanceSplitter implements DataSplitter {

	private InstanceAccessor dataModel;

	private IntegerArray trainReference;

	private IntegerArray testReference;

	public GivenInstanceSplitter(InstanceAccessor model, DataSelector selector) {
		this.dataModel = model;

		this.trainReference = new IntegerArray();
		this.testReference = new IntegerArray();
		int position = 0;
		for (DataInstance instance : model) {
			if (selector.select(instance)) {
				testReference.associateData(position++);
			} else {
				trainReference.associateData(position++);
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
