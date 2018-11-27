package com.jstarcraft.module.log;

import java.util.HashMap;
import java.util.Map;

/**
 * 游戏节点
 * 
 * <pre>
 * 行动与奖励和代价的组合称为节点
 * </pre>
 * 
 * @author Birdy
 *
 * @param <T>
 */
// TODO 此类可能重构为游戏日志工具
public class GameNode<T> {

	private T action;

	private Map<T, Integer> rewards = new HashMap<>();

	private Map<T, Integer> costs = new HashMap<>();

	GameNode(T action) {
		this.action = action;
	}

	public void putReward(T rewardId, int rewardNumber) {
		Integer value = rewards.get(rewardId);
		if (value == null) {
			value = rewardNumber;
		} else {
			value += rewardNumber;
		}
		rewards.put(rewardId, value);
	}

	public void putCost(T costId, int costNumber) {
		Integer value = costs.get(costId);
		if (value == null) {
			value = costNumber;
		} else {
			value += costNumber;
		}
		costs.put(costId, value);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int hash = 1;
		hash = prime * hash + action.hashCode();
		hash = prime * hash + ((costs == null) ? 0 : costs.size());
		hash = prime * hash + ((rewards == null) ? 0 : rewards.size());
		return hash;
	}

	@Override
	public boolean equals(Object object) {
		if (this == object)
			return true;
		if (object == null)
			return false;
		if (getClass() != object.getClass())
			return false;
		GameNode that = (GameNode) object;
		if (!this.action.equals(that.action))
			return false;
		if (this.costs == null) {
			if (that.costs != null)
				return false;
		} else if (!this.costs.equals(that.costs))
			return false;
		if (this.rewards == null) {
			if (that.rewards != null)
				return false;
		} else if (!this.rewards.equals(that.rewards))
			return false;
		return true;
	}

}
