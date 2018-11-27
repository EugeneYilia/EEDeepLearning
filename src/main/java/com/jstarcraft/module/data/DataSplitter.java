package com.jstarcraft.module.data;

import com.jstarcraft.module.data.accessor.InstanceAccessor;

/**
 * 数据处理器
 * 
 * @author Birdy
 *
 */
// TODO 负责将行为模型调整为训练模型+标记,测试模型+标记.
public interface DataSplitter {

	/**
	 * 获取分割数量
	 * 
	 * @return
	 */
	int getSize();

	/**
	 * 获取数据模型
	 * 
	 * @return
	 */
	InstanceAccessor getDataModel();

	/**
	 * 获取训练引用
	 * 
	 * @param index
	 * @return
	 */
	IntegerArray getTrainReference(int index);

	/**
	 * 获取测试引用
	 * 
	 * @param index
	 * @return
	 */
	IntegerArray getTestReference(int index);

}
