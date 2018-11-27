package com.jstarcraft.module.log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.jstarcraft.core.utility.JsonUtility;
import com.jstarcraft.core.utility.KeyValue;

public class JingLingZongDongYuanHandler {

	private static final Logger log = LoggerFactory.getLogger(JingLingZongDongYuanHandler.class);

	@Test
	public void test() throws Exception {
		Collection<File> files;
		File dataPath = new File("data/jing-ling-zong-dong-yuan/log");
		if (dataPath.isDirectory()) {
			files = FileUtils.listFiles(dataPath, null, true);
		} else {
			files = Arrays.asList(dataPath);
		}

		// 随机节点指向变换的行动.
		HashSet<Integer> randomActions = new HashSet<>();
		HashMap<Integer, KeyValue<Integer, Integer>> randomNodes = new HashMap<>();
		// 行动数据
		File actionFile = new File("data/jing-ling-zong-dong-yuan/action.txt");
		FileUtils.deleteQuietly(actionFile);
		actionFile.createNewFile();
		// 用户-等级映射(用于按照等级过滤)
		HashMap<Number, Integer> user2Levels = new HashMap<>();
		for (File file : files) {
			try (FileReader reader = new FileReader(file); BufferedReader in = new BufferedReader(reader)) {
				String line = null;
				while ((line = in.readLine()) != null) {
					Map<String, Object> data = JsonUtility.string2Object(line, Map.class);
					Number userId = (Number) data.get("uid");
					Integer level = (Integer) data.get("lv");
					user2Levels.put(userId, level);

					Integer action = (Integer) data.get("acid");
					if (randomActions.contains(action)) {
						continue;
					}
					List<Map<String, Object>> rewards = (List) data.get("revs");
					List<Map<String, Object>> costs = (List) data.get("coss");
					// 利用随机行为总是产生不定长度的特性.
					KeyValue<Integer, Integer> keyValue = randomNodes.get(action);
					if (keyValue == null) {
						keyValue = new KeyValue<>(rewards.size(), costs.size());
						randomNodes.put(action, keyValue);
					} else {
						if (keyValue.getKey() != rewards.size() || keyValue.getValue() != costs.size()) {
							randomActions.add(action);
						}
					}
				}
			}
		}
		// 节点数据
		File nodeFile = new File("data/jing-ling-zong-dong-yuan/node.txt");
		FileUtils.deleteQuietly(nodeFile);
		nodeFile.createNewFile();
		HashMap<GameNode, Integer> node2Ids = new HashMap<>(100000);
		HashMap<Integer, Integer> action2Numbers = new HashMap<>(100000);
		try (FileWriter actionWriter = new FileWriter(actionFile); BufferedWriter actionOut = new BufferedWriter(actionWriter); FileWriter nodeWriter = new FileWriter(nodeFile); BufferedWriter nodeOut = new BufferedWriter(nodeWriter);) {
			for (File file : files) {
				log.info(file.getName());
				try (FileReader reader = new FileReader(file); BufferedReader in = new BufferedReader(reader)) {
					String line = null;
					while ((line = in.readLine()) != null) {
						Map<String, Object> data = JsonUtility.string2Object(line, Map.class);
						Number userId = (Number) data.get("uid");
						// 等级限制
						if (user2Levels.get(userId) < 10) {
							continue;
						}
						Integer action = (Integer) data.get("acid");
						if (randomActions.contains(action)) {
							continue;
						}
						List<Map<String, Object>> rewards = (List) data.get("revs");
						List<Map<String, Object>> costs = (List) data.get("coss");
						// 交易限制
						GameNode node = new GameNode(action);
						if (costs != null && costs.size() > 0) {
							for (Map<String, Object> map : costs) {
								node.putCost((Integer) map.get("ciid"), (Integer) map.get("calter"));
							}
						} else {
							continue;
						}
						if (rewards != null && rewards.size() > 0) {
							for (Map<String, Object> map : rewards) {
								node.putReward((Integer) map.get("riid"), (Integer) map.get("ralter"));
							}
						} else {
							continue;
						}
						Integer nodeId = node2Ids.get(node);
						if (nodeId == null) {
							nodeId = node2Ids.size();
							node2Ids.put(node, nodeId);
							nodeOut.write(nodeId + " " + JsonUtility.object2String(node));
							nodeOut.newLine();

							Integer number = action2Numbers.get(action);
							if (number == null) {
								action2Numbers.put(action, 1);
							} else {
								action2Numbers.put(action, number + 1);
							}
						}

						Long instant = (Long) data.get("at");
						instant = TimeUnit.DAYS.convert(instant, TimeUnit.MILLISECONDS);
						Integer level = (Integer) data.get("lv");

						StringBuffer buffer = new StringBuffer(1000);
						buffer.append(userId).append(" ").append(nodeId).append(" ").append(instant).append(" ").append(level);
						actionOut.write(buffer.toString());
						actionOut.newLine();
						actionOut.flush();
					}
				}
				log.info("node size is {}", node2Ids.size());
			}
		}

		for (Entry<Integer, Integer> term : action2Numbers.entrySet()) {
			if (term.getValue() > 250) {
				log.info("term is {}:{}", new Object[] { term.getKey(), term.getValue() });
			}
		}
	}

}
