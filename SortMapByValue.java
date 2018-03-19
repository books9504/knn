import java.util.Comparator;
import java.util.HashMap;
import java.util.TreeMap;

public class SortMapByValue {
 
	public static void main(String[] args) {
		// <String, Integer> Map
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		map.put("a", 10);
		map.put("b", 30);
		map.put("c", 50);
		map.put("d", 40);
		map.put("e", 20);
		System.out.println(map);
 
 
		Comparator<String> comparator = new ValueComparator<String, Integer>(map);
		TreeMap<String, Integer> result = new TreeMap<String, Integer>(comparator);
		result.putAll(map);
 
		System.out.println(result);
 
		// <Integer, Integer> Map
 
		HashMap<Integer, Integer> map2 = new HashMap<Integer, Integer>();
		map2.put(1, 10);
		map2.put(2, 30);
		map2.put(3, 50);
		map2.put(4, 40);
		map2.put(5, 20);
		System.out.println(map2);
 
		Comparator<Integer> comparator2 = new ValueComparator<Integer, Integer>(map2);
		TreeMap<Integer, Integer> result2 = new TreeMap<Integer, Integer>(comparator2);
		result2.putAll(map2);
 
		System.out.println(result2);
 
	}
 
}
class ValueComparator<K, V extends Comparable<V>> implements Comparator<K>{
	 
	HashMap<K, V> map = new HashMap<K, V>();
 
	public ValueComparator(HashMap<K, V> map){
		this.map.putAll(map);
	}
 
	@Override
	public int compare(K s1, K s2) {
		return map.get(s1).compareTo(map.get(s2));//ascending order, put - for descending	
	}
}