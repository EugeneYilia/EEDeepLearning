package com.jstarcraft.module.log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.junit.Test;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

public class ExtractTransformLoad {

	private LinkedList<String> convert(CSVRecord values) {
		LinkedList<String> lines = new LinkedList<>();
		lines.add(values.get(0) + " " + values.get(1) + " 1 " + values.get(2) + "000");
		return lines;
	}

	@Test
	public void test() throws Exception {
		Collection<File> files;
		File dataPath = new File("data/hmm/user_item_time.txt");
		if (dataPath.isDirectory()) {
			files = FileUtils.listFiles(dataPath, null, true);
		} else {
			files = Arrays.asList(dataPath);
		}

		File memorandumFile = new File("data/hmm/user_item_time_100.txt");
		FileUtils.deleteQuietly(memorandumFile);
		memorandumFile.createNewFile();
		try (FileWriter writer = new FileWriter(memorandumFile); BufferedWriter out = new BufferedWriter(writer);) {
			for (File file : files) {
				try (FileReader reader = new FileReader(file); BufferedReader in = new BufferedReader(reader)) {
					try (CSVParser parser = new CSVParser(in, CSVFormat.newFormat('\t'))) {
						Iterator<CSVRecord> iterator = parser.iterator();
						while (iterator.hasNext()) {
							CSVRecord values = iterator.next();
							for (String line : convert(values)) {
								out.write(line);
								out.newLine();
							}
						}
					}
				}
			}
		}
	}

	@Test
	public void testRepeat() throws Exception {
		Collection<File> files;
		File dataPath = new File("data/poi/FourSquare/checkin");
		if (dataPath.isDirectory()) {
			files = FileUtils.listFiles(dataPath, null, true);
		} else {
			files = Arrays.asList(dataPath);
		}

		Table<String, String, String> tabel = HashBasedTable.create();

		File memorandumFile = new File("data/poi/FourSquare/checkin/norepeat.txt");
		FileUtils.deleteQuietly(memorandumFile);
		memorandumFile.createNewFile();
		try (FileWriter writer = new FileWriter(memorandumFile); BufferedWriter out = new BufferedWriter(writer);) {
			for (File file : files) {
				try (FileReader reader = new FileReader(file); BufferedReader in = new BufferedReader(reader)) {
					try (CSVParser parser = new CSVParser(in, CSVFormat.newFormat(' '))) {
						Iterator<CSVRecord> iterator = parser.iterator();
						while (iterator.hasNext()) {
							CSVRecord values = iterator.next();
							if (tabel.contains(values.get(0), values.get(1))) {
								continue;
							} else {
								tabel.put(values.get(0), values.get(1), values.get(2));
								out.write(values.get(0) + " " + values.get(1) + " " + values.get(2));
								out.newLine();
							}
						}
					}
				}
			}
		}
	}

}
