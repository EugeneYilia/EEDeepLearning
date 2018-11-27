package com.jstarcraft.module.log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.jstarcraft.core.utility.JsonUtility;

public class WuShuangTuLongHandler {

	private static final Logger log = LoggerFactory.getLogger(WuShuangTuLongHandler.class);

	@Test
	public void test() throws Exception {
		Collection<File> files;
		File dataPath = new File("data/wu-shuang-tu-long/goods/");
		if (dataPath.isDirectory()) {
			files = FileUtils.listFiles(dataPath, null, true);
		} else {
			files = Arrays.asList(dataPath);
		}

		// 行动数据
		File actionFile = new File("data/wu-shuang-tu-long/action.txt");
		FileUtils.deleteQuietly(actionFile);
		actionFile.createNewFile();
		HashMap<String, Integer> name2Indexes = new HashMap<>();
		name2Indexes.put("instant", 5);
		name2Indexes.put("user", 7);
		name2Indexes.put("action", 8);
		name2Indexes.put("item", 9);
		name2Indexes.put("number", 10);
		name2Indexes.put("change", 12);
		name2Indexes.put("level", 15);
		HashMap<GameNode, Integer> node2Ids = new HashMap<>();
		try (FileWriter writer = new FileWriter(actionFile); BufferedWriter out = new BufferedWriter(writer);) {
			for (File file : files) {
				log.info(file.getName());
				try (FileReader reader = new FileReader(file); BufferedReader in = new BufferedReader(reader)) {
					try (CSVParser parser = new CSVParser(in, CSVFormat.newFormat('|'))) {
						Iterator<CSVRecord> iterator = parser.iterator();
						while (iterator.hasNext()) {
							CSVRecord values = iterator.next();
							String instant = values.get(name2Indexes.get("instant"));
							String action = values.get(name2Indexes.get("action"));
							GameNode node = new GameNode(action);
							String item = values.get(name2Indexes.get("item"));
							Integer number = Integer.valueOf(values.get(name2Indexes.get("number")));
							if (values.get(name2Indexes.get("change")).equals("1")) {
								node.putReward(item, number);
								continue;
							} else {
								number = -number;
								node.putCost(item, number);
							}

							Integer nodeId = node2Ids.get(node);
							if (nodeId == null) {
								nodeId = node2Ids.size();
								node2Ids.put(node, nodeId);
							}

							// user node score instant level action item number
							String user = values.get(name2Indexes.get("user"));
							String level = values.get(name2Indexes.get("level"));
							String line = user + " " + nodeId + " 1 " + instant + " " + level + " " + action + " " + item + " " + number;

							out.write(line);
							out.newLine();
						}
					}
				}
			}
		}

		// 节点数据
		File nodeFile = new File("data/wu-shuang-tu-long/node.txt");
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
