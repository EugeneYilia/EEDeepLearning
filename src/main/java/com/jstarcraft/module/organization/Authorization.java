package com.jstarcraft.module.organization;

import com.jstarcraft.module.organization.configuration.Operation;

/**
 * 授权
 * 
 * @author Birdy
 *
 */
public class Authorization {

	/** 角色标识 */
	private Integer role;

	/** 资源标识 */
	private Long resource;

	/** 资源属性 */
	private String property;

	/** 操作 */
	private Operation operation;

}
