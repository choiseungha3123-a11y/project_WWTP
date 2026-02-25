package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Temporal;
import jakarta.persistence.TemporalType;
import jakarta.persistence.Transient;
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
public class PublicDTO {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private long no;
	@ManyToOne(fetch = FetchType.LAZY)
	@JsonIgnore
	//@JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
	@JoinColumn(name="userNo")
	Member member;
	@Column(name="userNo", insertable = false, updatable = false)
	private Long userNo;
	@Transient
	private String userName;
	@JsonIgnore
	private String userAgent;
	@JsonIgnore
	private String remoteInfo;
	private String pos;
	private String content;
	@JsonProperty(access = Access.WRITE_ONLY)
	private String password;
	@Temporal(TemporalType.TIMESTAMP)
	@JsonIgnore
	@Column(updatable = false)
	@Builder.Default
	private LocalDateTime createTime = LocalDateTime.now();
	@JsonIgnore
	private LocalDateTime modifyTime;
	//@JsonIgnore
	private LocalDateTime deleteTime;
	@Lob
    @Column(columnDefinition = "MEDIUMBLOB")
	private byte[] picture;
}
