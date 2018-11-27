package com.jstarcraft.module.data.processor;

import com.jstarcraft.module.data.accessor.DataInstance;

/**
 * 数据选择器
 * 
 * @author Birdy
 *
 */
public interface DataSelector {

	boolean select(DataInstance instance);

}
