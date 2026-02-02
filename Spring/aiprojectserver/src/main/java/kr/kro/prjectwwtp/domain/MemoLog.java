package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Temporal;
import jakarta.persistence.TemporalType;
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
public class MemoLog {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Schema(description = "고유번호", example = "1~")
	private long log_no;
	@Schema(description = "로그 종류", example = "list || create || modify || disable || oldlist")
	private String type;
	@Schema(description = "조회시의 페이지 번호", example = "1~")
	private int page;
	@Schema(description = "조회시의 페이지당 메모수", example = "10")
	private int count;
	@Schema(description = "수정, 종료한 메모의 고유번호", example = "1~")
	private long memoNo;
	@Schema(description = "수정 전 메모내용", example = "수정 전 메모내용")
	private String preContent;
	@Schema(description = "수정 후 메모내용", example = "수정 후 메모내용")
	private String currentContent;
	@ManyToOne(fetch = FetchType.LAZY)
	@JsonProperty(access = Access.WRITE_ONLY)
	@JoinColumn(name="userNo")
	@Schema(description = "요청 회원", example = "1~")
	private Member member;
	@Column(name="userNo", insertable = false, updatable = false)
    private Long userNo;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(updatable = false)
	@Builder.Default
	@Schema(description = "로그 생성 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime logTime = LocalDateTime.now();
}
