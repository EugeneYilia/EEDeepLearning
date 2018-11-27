package com.jstarcraft.module.organization.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import com.jstarcraft.core.utility.StringUtility;
import com.jstarcraft.module.organization.Resource;
import com.jstarcraft.module.organization.configuration.Operation;
import com.jstarcraft.module.organization.configuration.PropertyConfiguration;

/**
 * 组织权限
 * 
 * <pre>
 * 作用于具体资源类型的注解,与{@link Resource},{@link OrganizationResource}和{@link PropertyConfiguration}配合实现资源的权限控制
 * </pre>
 * 
 * @author Birdy
 *
 */
@Target({ ElementType.METHOD, ElementType.PARAMETER })
@Retention(RetentionPolicy.RUNTIME)
public @interface OrganizationPermission {

	/** 资源属性 */
	String property() default StringUtility.EMPTY;

	/** 资源操作 */
	Operation operation();

}
