package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.Map;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;

import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.evaluator.ranking.AUCEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.MAPEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NDCGEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NoveltyEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.PrecisionEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.RecallEvaluator;
import com.jstarcraft.module.recommendation.task.RankingTask;
import com.jstarcraft.module.recommendation.evaluator.ranking.MRREvaluator;

public class WBPRTestCase {

	@Test
	public void testRecommender() throws Exception {
		Configuration configuration = Configuration.valueOf("rec/collaborative/ranking/wbpr-test.properties");
		RankingTask job = new RankingTask(configuration);
		Map<String, Float> measures = job.execute();
		Assert.assertThat(measures.get(AUCEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.8143423F));
		Assert.assertThat(measures.get(MAPEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.2857075F));
		Assert.assertThat(measures.get(MRREvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.38666466F));
		Assert.assertThat(measures.get(NDCGEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.34714293F));
		Assert.assertThat(measures.get(NoveltyEvaluator.class.getSimpleName()), CoreMatchers.equalTo(16.477526F));
		Assert.assertThat(measures.get(PrecisionEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.2664537F));
		Assert.assertThat(measures.get(RecallEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.378611F));
	}

}
