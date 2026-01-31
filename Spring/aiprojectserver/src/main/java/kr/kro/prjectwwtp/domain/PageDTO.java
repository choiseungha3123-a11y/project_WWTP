package kr.kro.prjectwwtp.domain;

import java.util.List;

import org.springframework.data.domain.Page;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@NoArgsConstructor
public class PageDTO<T> {
	private int pageNo;
	private int numOfRows;
	private String type;
	private List<T> items;
	private long totalCount;
	private int totalPage;
	private boolean isLast;
	private boolean isFirst;
	private boolean isEmpty;
	
	public PageDTO(Page<T> page) {
		this.pageNo = page.getPageable().getPageNumber() + 1;
		this.numOfRows = page.getNumberOfElements();
		this.items = page.getContent();
		this.totalCount = page.getTotalElements();
		this.totalPage = page.getTotalPages();
		this.isLast = page.isLast();
		this.isFirst= page.isFirst();
		this.isEmpty = page.isEmpty();
	}
}
