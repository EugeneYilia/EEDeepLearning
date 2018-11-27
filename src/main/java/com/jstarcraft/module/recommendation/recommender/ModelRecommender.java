package com.jstarcraft.module.recommendation.recommender;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.exception.RecommendationException;

/**
 * 模型推荐器
 * 
 * <pre>
 * 与机器学习相关
 * </pre>
 * 
 * @author Birdy
 *
 */
public abstract class ModelRecommender extends AbstractRecommender {

	/** 周期次数 */
	protected int numberOfEpoches;

	/** 是否收敛(early-stop criteria) */
	protected boolean isConverged;

	/** 用于观察损失率 */
	protected float totalLoss, currentLoss = 0F;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// 参数部分
		numberOfEpoches = configuration.getInteger("rec.iterator.maximum", 100);
		isConverged = configuration.getBoolean("rec.recommender.earlystop", false);
	}

	/**
	 * 是否收敛
	 * 
	 * @param iteration
	 * @return
	 */
	protected boolean isConverged(int iteration) {
		float deltaError = currentLoss - totalLoss;
		// print out debug info
		if (logger.isInfoEnabled()) {
			String recName = getClass().getSimpleName();
			String info = recName + " iter " + iteration + ": loss = " + totalLoss + ", delta_loss = " + deltaError;
			logger.info(info);
		}
		if (Float.isNaN(totalLoss) || Float.isInfinite(totalLoss)) {
			// LOG.error("Loss = NaN or Infinity: current settings does not fit
			// the recommender! Change the settings and try again!");
			throw new RecommendationException("Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!");
		}
		// check if converged
		boolean converged = Math.abs(deltaError) < MathUtility.EPSILON;
		return converged;
	}

	/**
	 * fajie To calculate cmg based on pairwise loss function type
	 * 
	 * @param lossType
	 * @param error
	 * @return
	 */
	protected final float calaculateGradientValue(int lossType, float error) {
		final float constant = 1F;
		float value = 0F;
		switch (lossType) {
		case 0:// Hinge loss
			if (constant * error <= 1F)
				value = constant;
			break;
		case 1:// Rennie loss
			if (constant * error <= 0F)
				value = -constant;
			else if (constant * error <= 1F)
				value = (1F - constant * error) * (-constant);
			else
				value = 0F;
			value = -value;
			break;
		case 2:// logistic loss, BPR
			value = MathUtility.logistic(-error);
			break;
		case 3:// Frank loss
			value = (float) (Math.sqrt(MathUtility.logistic(error)) / (1F + Math.exp(error)));
			break;
		case 4:// Exponential loss
			value = (float) Math.exp(-error);
			break;
		case 5:// quadratically smoothed
			if (error <= 1F)
				value = 0.5F * (1F - error);
			break;
		default:
			break;
		}
		return value;
	}

}
