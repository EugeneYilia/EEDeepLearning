package com.jstarcraft.module.recommendation.recommender.context.ranking;

import java.util.Map;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;

import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.evaluator.ranking.AUCEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.MAPEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.MRREvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NDCGEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NoveltyEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.PrecisionEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.RecallEvaluator;
import com.jstarcraft.module.recommendation.task.RankingTask;

public class SBPRTestCase {

	@Test
	public void testRecommender() throws Exception {
		Configuration configuration = Configuration.valueOf("rec/context/ranking/sbpr-test.properties");
		RankingTask job = new RankingTask(configuration);
		Map<String, Float> measures = job.execute();
		Assert.assertThat(measures.get(AUCEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.9311945F));
		Assert.assertThat(measures.get(MAPEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.45850572F));
		Assert.assertThat(measures.get(MRREvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.6168819F));
		Assert.assertThat(measures.get(NDCGEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.5544954F));
		Assert.assertThat(measures.get(NoveltyEvaluator.class.getSimpleName()), CoreMatchers.equalTo(16.66047F));
		Assert.assertThat(measures.get(PrecisionEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.34571743F));
		Assert.assertThat(measures.get(RecallEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.6222653F));
	}

}
