package com.jstarcraft.module.organization.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import com.jstarcraft.module.organization.configuration.Relation;

/**
 * 组织资源
 * 
 * <pre>
 * 与{@link OrganizationPermission}配合
 * </pre>
 * 
 * @author Administrator
 *
 */
@Target({ ElementType.TYPE })
@Retention(RetentionPolicy.RUNTIME)
public @interface OrganizationResource {

	/** 资源类型 */
	String type();

	/** 资源属性(TODO 考虑基于静态注解或者基于动态配置) */
	String[] properties() default {};

	/** 资源关系(资源与包含此资源的分组之间的关系) */
	Relation relation() default Relation.AGGREGATION;

}
