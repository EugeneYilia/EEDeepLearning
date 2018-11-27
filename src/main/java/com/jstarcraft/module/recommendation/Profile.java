package com.jstarcraft.module.recommendation;

import java.util.Map;

/**
 * 剖面
 * 
 * @author Birdy
 *
 */
public interface Profile {

	/**
	 * 创建特征
	 * 
	 * @param feature
	 * @param score
	 * @return
	 */
	boolean createFeature(String feature, int score);

	/**
	 * 修改特征
	 * 
	 * @param feature
	 * @param score
	 * @return
	 */
	boolean updateFeature(String feature, int score);

	/**
	 * 删除特征
	 * 
	 * @param feature
	 * @return
	 */
	boolean deleteFeature(String feature);

	/**
	 * 获取特征
	 * 
	 * @return
	 */
	Map<String, Integer> getFeatures();

}
