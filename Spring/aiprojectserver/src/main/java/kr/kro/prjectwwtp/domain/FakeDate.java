package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

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
public class FakeDate {
	@Id
	@Schema(description = "임시 날짜를 조회한 오늘 날짜", example = "2026-01-30T15:04:05")
	private LocalDateTime today;
	@Schema(description = "임시 날짜", example = "2026-01-15")
	private LocalDateTime fakeDate;
}
