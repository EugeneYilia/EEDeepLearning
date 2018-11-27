package com.jstarcraft.module.organization;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

import com.jstarcraft.module.organization.annotation.OrganizationPermission;
import com.jstarcraft.module.organization.configuration.Operation;

/**
 * 组
 * 
 * @author Birdy
 *
 */
public class Group implements Resource {

	/** 标识 */
	private Long id;

	/** 名称(update TODO 聚合) */
	private String name;

	/** 所属组(update TODO 聚合) */
	private Long group;

	/** 所属人(update TODO 聚合) */
	private Long owner;

	/** 分组(create,delete,retrieve TODO 组合) */
	private Collection<Long> groups;

	/** 授权(update TODO 聚合) */
	private Collection<Authorization> authorizations;

	/** 资源标识-类型映射(create,update,delete,retrieve TODO 聚合) */
	private Map<Serializable, String> resources;

	/** 角色(create,update,delete,retrieve TODO 聚合) */
	private Map<Integer, String> roles;

	/** 用户(create,delete,retrieve TODO 组合) */
	private Collection<Long> users;

	@Override
	public Long getId() {
		return id;
	}

	@Override
	@OrganizationPermission(property = "name", operation = Operation.RETRIEVE)
	public String getName() {
		return name;
	}
	
	@OrganizationPermission(property = "name", operation = Operation.UPDATE)
	public void setName(String name) {
		// TODO
	}

	@Override
	public Long getOwner() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Long getGroup() {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * 创建指定的角色
	 * 
	 * @param name
	 * @return
	 */
	@OrganizationPermission(property = "roles", operation = Operation.CREATE)
	public Integer createRole(String name) {
		return null;
	}

	/**
	 * 删除指定的角色
	 * 
	 * @param name
	 * @return
	 */
	@OrganizationPermission(property = "roles", operation = Operation.DELETE)
	public boolean deleteRole(Integer id) {
		return false;
	}

	/**
	 * 获取所有的角色
	 * 
	 * @return
	 */
	@OrganizationPermission(property = "roles", operation = Operation.RETRIEVE)
	public Map<Integer, String> getRoles() {
		return null;
	}

	/**
	 * 修改指定的角色
	 * 
	 * @return
	 */
	@OrganizationPermission(property = "roles", operation = Operation.UPDATE)
	public boolean changeRole(Integer id, String name) {
		return false;
	}

	/**
	 * 创建指定的授权
	 * 
	 * @param name
	 * @return
	 */
	@OrganizationPermission(property = "authorizations", operation = Operation.CREATE)
	public Authorization createAuthorization(Integer role, Serializable resource, String property, Operation operation) {
		return null;
	}

	/**
	 * 删除指定的授权
	 * 
	 * @param name
	 * @return
	 */
	@OrganizationPermission(property = "authorizations", operation = Operation.DELETE)
	public boolean deleteAuthorization(Authorization authorization) {
		return false;
	}

	/**
	 * 获取角色的授权
	 * 
	 * @return
	 */
	@OrganizationPermission(property = "authorizations", operation = Operation.RETRIEVE)
	public Collection<Authorization> getAuthorizations4Role(Integer id) {
		return null;
	}

	/**
	 * 获取资源的授权
	 * 
	 * @return
	 */
	@OrganizationPermission(property = "authorizations", operation = Operation.RETRIEVE)
	public Collection<Authorization> getAuthorizations4Resource(Serializable id) {
		return null;
	}

	/**
	 * 创建指定的资源
	 * 
	 * @param name
	 * @return
	 */
	public boolean createResource(@OrganizationPermission(operation = Operation.CREATE) String type, Serializable id) {
		return false;
	}

	/**
	 * 删除指定的资源
	 * 
	 * @param name
	 * @return
	 */
	public boolean deleteResource(@OrganizationPermission(operation = Operation.DELETE) String type, Serializable id) {
		return false;
	}

	/**
	 * 检索指定类型的资源
	 * 
	 * @return
	 */
	public Collection<Serializable> getResources(@OrganizationPermission(operation = Operation.RETRIEVE) String type) {
		return null;
	}

}
