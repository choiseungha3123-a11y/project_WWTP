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

/**
 * 메모 전송용 DTO
 * - 화면(또는 API) <-> 컨트롤러 간에 메모 데이터를 주고받을 때 사용합니다.
 */
@Getter
@Setter
@ToString
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Entity
public class Memo{
    @Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
    private long memoNo;
    private String content;
    
    @ManyToOne(fetch = FetchType.LAZY)
    //@JsonProperty(access = Access.WRITE_ONLY)
    @JoinColumn(name="createUserNo")
    @Schema(description = "메모의 최초 생성 회원", example = "1~")
    private Member createMember;
    //@Column(name="createUserNo", insertable = false, updatable = false)
    //private Long createUserNo;
    @ManyToOne(fetch = FetchType.LAZY)
    //@JsonProperty(access = Access.WRITE_ONLY)
    @JoinColumn(name="lastModifyUserNo")
    @Schema(description = "메모를 수정한 회원", example = "1~")
    private Member modifyMember;
    //@Column(name="lastModifyUserNo", insertable = false, updatable = false)
    //private Long lastModifyUserNo;
    @ManyToOne(fetch = FetchType.LAZY)
    //@JsonProperty(access = Access.WRITE_ONLY)
    @JoinColumn(name="diableUseNo")
    @Schema(description = "메모를 종료한 회원", example = "1~")
    private Member disableMember;
    //@Column(name="diableUseNo", insertable = false, updatable = false)
    //private Long diableUseNo;
    
    @Temporal(TemporalType.TIMESTAMP)
	@Column(updatable = false)
	@Builder.Default
	@Schema(description = "메모 생성 시간", example = "2026-01-30T15:30:00")
	LocalDateTime createTime = LocalDateTime.now();
    @Temporal(TemporalType.TIMESTAMP)
    @Schema(description = "메모 종료 처리 시간", example = "2026-01-30T15:30:00")
	LocalDateTime disableTime;
}
