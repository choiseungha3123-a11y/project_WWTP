package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
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
public class Member {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Schema(description = "고유번호", example = "1~")
	private long userNo;
	@Schema(description = "ID", example = "user123")
	private String userId;
	@Schema(description = "회원이름", example = "회원이름")
	private String userName;
	@Schema(description = "회원Email", example = "회원Email")
	private String userEmail;
	@JsonProperty(access = Access.WRITE_ONLY)
	@Schema(description = "비밀번호", example = "비밀번호는 10~20자이며, 영문 대/소문자, 숫자, 특수문자를 각각 1개 이상 포함해야 합니다.")
	private String password;
	@Enumerated(EnumType.STRING)
	@Schema(description = "회원권한", example = "ROLE_ADNIN || ROLE_MEMBER || ROLE_VIEWER")
	private Role role;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(updatable = false)
	@Builder.Default
	@Schema(description = "생성 시간", example = "2026-01-30T15:30:00")
	private LocalDateTime createTime = LocalDateTime.now(); 
	@Temporal(TemporalType.TIMESTAMP)
	@Builder.Default
	private LocalDateTime lastLoginTime = LocalDateTime.now();
	@Schema(description = "마지막 로그인 시간", example = "2026-01-30T15:30:00")
	private String socialAuth;
}
