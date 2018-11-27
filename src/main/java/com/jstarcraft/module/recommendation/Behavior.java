package com.jstarcraft.module.recommendation;

/**
 * 行为日志
 * 
 * @author Birdy
 *
 */
public class Behavior {

	/** 用户标识 */
	private long userId;

	/** 物品标识 */
	private long itemId;

	/** 类型 */
	private String type;

	/** 内容 */
	private Object content;

	/** 上下文 */
	private Object context;

	/** 权重 */
	private double weight;

}
