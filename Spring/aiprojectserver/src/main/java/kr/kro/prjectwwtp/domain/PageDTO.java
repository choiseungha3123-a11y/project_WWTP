package kr.kro.prjectwwtp.domain;

import java.util.List;

import org.springframework.data.domain.Page;

import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@NoArgsConstructor
public class PageDTO<T> {
	@Schema(description = "조회된 페이지 번호", example = "1")
	private int pageNo;
	@Schema(description = "결과 수", example = "10")
	private int numOfRows;
//	@Schema(description = "추후에 사용하려는 구분", example = "NULL || TYPE1 || TYPE2")
//	private String type;
	@ArraySchema(schema = @Schema(description = "조회한 Object List", example = "[]"))
	private List<T> items;
	@Schema(description = "조회 가능한 전체 수", example = "1234")
	private long totalCount;
	@Schema(description = "조회 가능한 전체 페이지수", example = "1")
	private int totalPage;
	@Schema(description = "마지막 페이지 인지 아닌지", example = "true || false")
	private boolean isLast;
	@Schema(description = "첫 페이지 인지 아닌지", example = "true || false")
	private boolean isFirst;
	@Schema(description = "조회된 데이터가 있는지 없는지", example = "true || false")
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
