package com.jstarcraft.module.organization;

/**
 * 资源
 * 
 * @author Birdy
 *
 */
public interface Resource {

	/**
	 * 获取资源标识
	 * 
	 * @return
	 */
	Long getId();

	/**
	 * 获取资源名称
	 * 
	 * @return
	 */
	String getName();

	/**
	 * 获取所属人
	 * 
	 * @return
	 */
	Long getOwner();

	/**
	 * 获取所属组
	 * 
	 * @return
	 */
	Long getGroup();

}
