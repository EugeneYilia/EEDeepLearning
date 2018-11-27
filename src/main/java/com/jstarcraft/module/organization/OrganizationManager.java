package com.jstarcraft.module.organization;

/**
 * 组织管理器
 * 
 * @author Birdy
 *
 */
public interface OrganizationManager {

	/**
	 * 获取指定的资源的权限代理
	 * 
	 * @param resource
	 * @param user
	 * @return
	 */
	<T extends Resource> T getProxy(T resource, User user);

}
