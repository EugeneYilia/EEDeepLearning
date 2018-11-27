package com.jstarcraft.module.log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.jstarcraft.core.utility.JsonUtility;

public class FengHuoJiuYouHandler {

	private static final Logger log = LoggerFactory.getLogger(FengHuoJiuYouHandler.class);

	@Test
	public void test() throws Exception {
		Collection<File> files;
		File dataPath = new File("data/feng-huo-jiu-you/log");
		if (dataPath.isDirectory()) {
			files = FileUtils.listFiles(dataPath, null, true);
		} else {
			files = Arrays.asList(dataPath);
		}

		// 行动数据
		File actionFile = new File("data/feng-huo-jiu-you/action.txt");
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
				}
			}
		}
		// 节点-标识映射
		HashMap<GameNode, Integer> node2Ids = new HashMap<>();
		try (FileWriter writer = new FileWriter(actionFile); BufferedWriter out = new BufferedWriter(writer);) {
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
						List<Map<String, Object>> rewards = (List) data.get("revs");
						List<Map<String, Object>> costs = (List) data.get("coss");
						// 交易限制
						GameNode node = new GameNode(action);
						if (costs != null) {
							for (Map<String, Object> map : costs) {
								node.putCost((Integer) map.get("ciid"), (Integer) map.get("calter"));
							}
						} else {
							continue;
						}

						if (rewards != null) {
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
						}

						Long instant = (Long) data.get("at");
						instant = TimeUnit.DAYS.convert(instant, TimeUnit.MILLISECONDS);
						Integer level = (Integer) data.get("lv");

						out.write(userId + " " + nodeId + " 1 " + instant + " " + level);
						out.newLine();
					}
				}
			}
		}

		// 节点数据
		File nodeFile = new File("data/feng-huo-jiu-you/node.txt");
		FileUtils.deleteQuietly(nodeFile);
		nodeFile.createNewFile();
		try (FileWriter writer = new FileWriter(nodeFile); BufferedWriter out = new BufferedWriter(writer);) {
			for (Entry<GameNode, Integer> term : node2Ids.entrySet()) {
				out.write(term.getValue() + " " + JsonUtility.object2String(term.getKey()));
				out.newLine();
			}
		}
	}

}
