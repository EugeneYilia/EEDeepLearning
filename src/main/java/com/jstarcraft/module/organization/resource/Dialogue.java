package com.jstarcraft.module.organization.resource;

import java.util.Collection;
import java.util.List;

import com.jstarcraft.module.organization.Resource;
import com.jstarcraft.module.organization.annotation.OrganizationPermission;
import com.jstarcraft.module.organization.configuration.Operation;

/**
 * 讨论
 * 
 * @author Birdy
 *
 */
public class Dialogue implements Resource {

	/** 标识 */
	private Long id;

	/** 名称(update TODO 聚合) */
	private String name;

	/** 用户(create,delete,retrieve) */
	private Collection<Long> users;

	/** 用户(update,retrieve TODO 组合) */
	private List<String> records;

	@Override
	public Long getId() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return null;
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

	@OrganizationPermission(property = "users", operation = Operation.CREATE)
	public boolean includeUser(Long user) {
		return false;
	}

	@OrganizationPermission(property = "users", operation = Operation.DELETE)
	public boolean excludeUser(Long user) {
		return false;
	}

	@OrganizationPermission(property = "users", operation = Operation.RETRIEVE)
	public Collection<Long> getUsers() {
		return null;
	}

	@OrganizationPermission(property = "records", operation = Operation.UPDATE)
	public void talk(String content) {

	}

	@OrganizationPermission(property = "records", operation = Operation.RETRIEVE)
	public List<String> getRecords(Integer from, Integer to) {
		return null;
	}

}
