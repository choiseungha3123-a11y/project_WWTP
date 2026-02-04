package kr.kro.prjectwwtp.domain;

import java.util.Date;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Entity
public class TmsSummary {
	@Id
	@Schema(description = "날짜", example = "2026-01-30")
	private Date time;
	@Schema(description = "데이터수", example = "0~")
	private int count;
	

}
