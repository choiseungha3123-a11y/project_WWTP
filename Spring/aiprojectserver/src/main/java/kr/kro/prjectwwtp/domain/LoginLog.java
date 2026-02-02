package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

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
public class LoginLog {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Schema(description = "고유번호", example = "1~")
	private long log_no;
	@ManyToOne(fetch = FetchType.LAZY)
	@JoinColumn(name="userNo")
	private Member member;
	@Column(name="userNo", insertable = false, updatable = false)
	@Schema(description = "접속 회원 고유번호", example = "0~ 0은 비회원")
    private Long userNo;
	@Schema(description = "로그인 시도한 ID")
	private String userId;
	@Schema(description = "로그인 성공 여부", example = "true / false")
	private boolean success;
	@Schema(description = "로그인 시도한 IP 주소")
	private String remoteInfo;
	@Schema(description = "오류 메시지", example = "처리중 발생한 오류")
	private String errorMsg;
	@Schema(description = "소셜로그인 정보")
	private String socialAuth;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(updatable = false)
	@Builder.Default
	@Schema(description = "로그 생성 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime logTime = LocalDateTime.now();
}
