package com.jstarcraft.module.text;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

public class Ngram<T> implements Iterable<List<T>> {

	private int grams;

	private List<T> terms;

	public Ngram(int grams, T... terms) {
		assert grams <= terms.length;
		this.grams = grams;
		this.terms = Arrays.asList(terms);
	}

	public Ngram(int grams, Collection<T> terms) {
		assert grams <= terms.size();
		this.grams = grams;
		this.terms = new ArrayList<>(terms);
	}

	public int getSize() {
		return terms.size() - grams + 1;
	}

	@Override
	public Iterator<List<T>> iterator() {
		return new NgramIterator();
	}

	private class NgramIterator implements Iterator<List<T>> {

		private int size = terms.size();

		private int cursor;

		@Override
		public boolean hasNext() {
			return cursor + grams <= size;
		}

		@Override
		public List<T> next() {
			List<T> ngram = terms.subList(cursor, cursor + grams);
			cursor++;
			return ngram;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

	}

}
