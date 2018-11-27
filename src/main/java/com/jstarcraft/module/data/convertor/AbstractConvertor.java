package com.jstarcraft.module.data.convertor;

import java.util.Map;

import com.jstarcraft.module.data.DataConvertor;

public abstract class AbstractConvertor<T> implements DataConvertor {

	/** 名称 */
	protected final String name;

	/** 字段 */
	protected final Map<String, T> fields;

	protected AbstractConvertor(String name, Map<String, T> fields) {
		this.name = name;
		this.fields = fields;
	}

	@Override
	public String getName() {
		return name;
	}

}
