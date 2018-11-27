package com.jstarcraft.module.organization.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * 基于角色的权限检查
 * 
 * <pre>
 * 与{@link NameParameter}配合实现权限控制
 * 当group,role为{}形式的字符串,会取对应命名参数的值设置到注解.
 * </pre>
 * 
 * @author Birdy
 *
 */
@Target({ ElementType.METHOD })
@Retention(RetentionPolicy.RUNTIME)
public @interface CheckRole {

	/** 角色组 */
	String group();

	/** 角色标识或者角色名称 */
	String role();

}
