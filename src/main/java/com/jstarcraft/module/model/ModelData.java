package com.jstarcraft.module.model;

/**
 * 模型数据
 * 
 * <pre>
 * 当数据为复杂类型时,valueData为ModelData[],size大于等于0.
 * 当数据为简单类型时,valueData为ModelData,size等于0
 * </pre>
 * 
 * @author Birdy
 *
 */
class ModelData {

	/**
	 * 类型(格式为JavaType)
	 */
	private byte[] keyData;

	/**
	 * 内容(格式由ContentCodec决定)
	 */
	private byte[] valueData;

	/**
	 * 数量
	 */
	private int size;

	ModelData() {
	}

	ModelData(byte[] keyData, byte[] valueData, int size) {
		this.keyData = keyData;
		this.valueData = valueData;
		this.size = size;
	}

	public byte[] getKeyData() {
		return keyData;
	}

	public byte[] getValueData() {
		return valueData;
	}

	public int getSize() {
		return size;
	}

}
