package com.jstarcraft.module.organization.configuration;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;

/**
 * 资源属性配置
 * 
 * @author Birdy
 *
 */
public class PropertyConfiguration {

	/** 资源类别 */
	private String type;

	/** 属性操作(名称-可分配的操作) */
	private Map<String, Collection<Operation>> properties;

	/** 资源关系(资源与包含此资源的分组之间的关系) */
	private Relation relation;

	public String getType() {
		return type;
	}

	public Collection<String> getProperties() {
		return Collections.unmodifiableCollection(properties.keySet());
	}

	public Collection<Operation> getOperations(String property) {
		return Collections.unmodifiableCollection(properties.get(property));
	}

	public Relation getRelation() {
		return relation;
	}

}