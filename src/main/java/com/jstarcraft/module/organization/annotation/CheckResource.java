package com.jstarcraft.module.organization.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import com.jstarcraft.module.organization.configuration.Operation;

/**
 * 基于资源的权限检查
 * 
 * <pre>
 * 与{@link NameParameter}配合实现权限控制
 * operation {type}/{resource}/{property}
 * 当type,resource,property为{}形式的字符串,会取对应命名参数的值设置到注解.
 * </pre>
 * 
 * @author Birdy
 *
 */
@Target({ ElementType.METHOD })
@Retention(RetentionPolicy.RUNTIME)
public @interface CheckResource {

	/** 资源操作 */
	Operation operation();

	/** 资源类型 */
	String type();

	/** 资源标识 */
	String resource();

	/** 资源属性 */
	String property();

}
