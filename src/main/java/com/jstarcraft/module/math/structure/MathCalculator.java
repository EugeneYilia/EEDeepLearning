package com.jstarcraft.module.math.structure;

import com.jstarcraft.module.math.structure.message.MessageStorage;

/**
 * 数学计算器
 * 
 * @author Birdy
 *
 */
public enum MathCalculator {

	/** 串行 */
	SERIAL,

	/** 并行 */
	PARALLEL;

	public final static ThreadLocal<MessageStorage> STORAGES = new ThreadLocal<>();

	// TODO 准备重构为buildStorage与getStorage.
	public static MessageStorage getStorage() {
		MessageStorage storage = STORAGES.get();
		if (storage == null) {
			storage = new MessageStorage();
			STORAGES.set(storage);
		}
		return storage;
	}

}
