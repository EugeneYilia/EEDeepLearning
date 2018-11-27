package com.jstarcraft.module.organization.resource;

import java.time.Instant;

import com.jstarcraft.module.organization.Resource;
import com.jstarcraft.module.organization.annotation.OrganizationPermission;
import com.jstarcraft.module.organization.configuration.Operation;

/**
 * 文章
 * 
 * @author Birdy
 *
 */
public class Article implements Resource {

	/** 标识 */
	private Long id;

	/** 标题 */
	private String title;

	/** 内容 */
	private String content;

	/** 修改时间 */
	private Instant updatedAt;

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

	@OrganizationPermission(property = "content", operation = Operation.RETRIEVE)
	public String getTitle() {
		return title;
	}

	@OrganizationPermission(property = "content", operation = Operation.UPDATE)
	public void setTitle(String title) {
		this.title = title;
	}

	@OrganizationPermission(property = "content", operation = Operation.RETRIEVE)
	public String getContent() {
		return content;
	}

	@OrganizationPermission(property = "content", operation = Operation.UPDATE)
	public void setContent(String content) {
		this.content = content;
	}

}
